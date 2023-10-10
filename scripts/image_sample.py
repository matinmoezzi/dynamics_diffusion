from datetime import datetime
import os
from functools import partial
from pathlib import Path
import pathlib
import time
from omegaconf import OmegaConf
import hydra
import numpy as np
import torch as th
import torch.distributed as dist

from dynamics_diffusion import dist_util, logger, sde_sampling
from dynamics_diffusion.karras_diffusion import karras_sample
from dynamics_diffusion.random_util import get_generator
from dynamics_diffusion.sde import VESDE
import matplotlib.pyplot as plt

NUM_CLASSES = 1000


def image_grid(x, image_size=32, num_channels=3):
    size = image_size
    channels = num_channels
    img = x.reshape(-1, size, size, channels)
    w = int(np.sqrt(img.shape[0]))
    img = (
        img.reshape((w, w, size, size, channels))
        .transpose((0, 2, 1, 3, 4))
        .reshape((w * size, w * size, channels))
    )
    return img


def show_samples(x):
    x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
    img = image_grid(x)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(img)
    plt.show()


def main():
    abs_path = Path(__file__).parent
    cfg = OmegaConf.load(abs_path / "../config/dynamics/sample_image.yaml")
    if cfg.model_dir.split(".")[-1] in ["pt", "pth"]:
        assert cfg.model_dir is not None or cfg.model_dir != "", "Model not found."
        model_path = cfg.model_dir

        assert (
            cfg.model_cfg is not None or cfg.model_cfg != ""
        ), "Model config not found."
        model_cfg = OmegaConf.load(abs_path / ".." / cfg.model_cfg)

        assert (
            cfg.diffusion_cfg is not None or cfg.diffusion_cfg != ""
        ), "Diffusion config not found."
        diffusion_cfg = OmegaConf.load(abs_path / ".." / cfg.diffusion_cfg)

        assert cfg.sampler is not None or cfg.sampler != "", "Invalid Sampler."
        sampler = cfg.sampler

        if sampler == "cm":
            cm_training_mode = cfg.cm_sampler.training_mode
        if sampler == "sde":
            continuous = model_cfg.target.continuous

    else:
        assert Path(cfg.model_dir, ".hydra").is_dir(), "Hydra configuration not found."

        model_prefix = "ema" if cfg.use_ema else "model"

        list_models = list(
            pathlib.Path(cfg.model_dir, "train").glob(f"{model_prefix}*.pt")
        )
        assert list_models, f"No {model_prefix} found."

        model_path = max(list_models, key=os.path.getctime)

        train_cfg = OmegaConf.load(Path(cfg.model_dir, ".hydra", "config.yaml"))

        model_cfg = train_cfg.trainer.model
        diffusion_cfg = train_cfg.trainer.diffusion

        if train_cfg.trainer._target_.split(".")[-1] == "DDPMImageTrainer":
            sampler = "ddpm"
        elif train_cfg.trainer._target_.split(".")[-1] == "CMImageTrainer":
            sampler = "cm"
            cm_training_mode = train_cfg.trainer.training_mode
        elif train_cfg.trainer._target_.split(".")[-1] == "SDEImageTrainer":
            sampler = "sde"
            continuous = train_cfg.trainer.continuous
        else:
            raise NotImplementedError(f"{train_cfg.trainer._target_} not supported.")

    log_dir = abs_path / f"../image_samples/{sampler}/{datetime.now():%Y%m%d-%H%M%S}"
    dist_util.setup_dist()
    logger.configure(dir=str(log_dir), format_strs=["stdout"])

    logger.log("creating model and diffusion...")

    model = hydra.utils.instantiate(model_cfg.target)
    if sampler in ["ddpm", "cm"]:
        model.load_state_dict(
            dist_util.load_state_dict(str(model_path), map_location="cpu")
        )

        model.to(dist_util.dev())

        if model_cfg.target.use_fp16:
            model.convert_to_fp16()

        model.eval()
    elif sampler == "sde":
        model = th.nn.DataParallel(model)
        model.to(dist_util.dev())
        loaded_state = dist_util.load_state_dict(
            str(model_path), map_location=dist_util.dev()
        )
        model.load_state_dict(loaded_state["model"], strict=False)

    model_kwargs = {}
    if sampler == "ddpm":
        diffusion = hydra.utils.call(diffusion_cfg.target)
        sample_fn = (
            diffusion.p_sample_loop
            if not cfg.ddpm_sampler.use_ddim
            else diffusion.ddim_sample_loop
        )
        if model_cfg.target.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(cfg.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn_wrapper = partial(
            sample_fn,
            model,
            (
                cfg.batch_size,
                3,
                model_cfg.target.image_size,
                model_cfg.target.image_size,
            ),
            clip_denoised=cfg.ddpm_sampler.clip_denoised,
            model_kwargs=model_kwargs,
        )
    elif sampler == "cm":
        assert cm_training_mode is not None, "CM training mode not found."
        if "consistency" in cm_training_mode:
            distillation = True
        else:
            distillation = False
        diffusion = hydra.utils.call(diffusion_cfg.target, distillation=distillation)
        if cfg.cm_sampler.sampler == "multistep":
            assert len(cfg.ts) > 0
            ts = tuple(int(x) for x in cfg.ts.split(","))
        else:
            ts = None
        generator = get_generator(
            cfg.cm_sampler.generator, cfg.num_samples, cfg.cm_sampler.seed
        )
        if model_cfg.target.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(cfg.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn_wrapper = partial(
            karras_sample,
            diffusion,
            model,
            (
                cfg.batch_size,
                3,
                model_cfg.target.image_size,
                model_cfg.target.image_size,
            ),
            steps=cfg.cm_sampler.steps,
            device=dist_util.dev(),
            clip_denoised=cfg.cm_sampler.clip_denoised,
            sampler=cfg.cm_sampler.sampler,
            sigma_min=diffusion_cfg.target.sigma_min,
            sigma_max=diffusion_cfg.target.sigma_max,
            s_churn=cfg.cm_sampler.s_churn,
            s_tmin=cfg.cm_sampler.s_tmin,
            s_tmax=cfg.cm_sampler.s_tmax,
            s_noise=cfg.cm_sampler.s_noise,
            generator=generator,
            ts=ts,
            model_kwargs=model_kwargs,
        )
    elif sampler == "sde":
        sde = hydra.utils.instantiate(diffusion_cfg.target)
        sampling_eps = 1e-3
        if isinstance(sde, VESDE):
            sampling_eps = 1e-5
        sampling_shape = (
            cfg.batch_size,
            3,
            model_cfg.target.image_size,
            model_cfg.target.image_size,
        )
        inverse_scaler = (
            lambda x: (x + 1.0) / 2.0 if model_cfg.target.data_centered else x
        )
        sampling_fn = partial(
            sde_sampling.get_sampling_fn,
            cfg.sde_sampler,
            sde,
            sampling_shape,
            inverse_scaler,
            sampling_eps,
            continuous=continuous,
            device=dist_util.dev(),
        )
    else:
        raise NotImplementedError(f"{sampler} not supported.")

    logger.log(f"sampling...")

    all_samples = []
    start = time.time()
    while len(all_samples) * cfg.batch_size < cfg.num_samples:
        if sampler == "sde":
            sample, n = sampling_fn()(model)
            show_samples(sample)
        else:
            sample = sample_fn_wrapper()
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

        logger.log(f"created {len(all_samples) * cfg.batch_size} samples")
    end = time.time()

    samples_arr = np.concatenate(all_samples, axis=0)[: cfg.num_samples]

    logger.log(f"{cfg.num_samples} sampled in {end - start:.4f} sec")

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in samples_arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, samples_arr)

    dist.barrier()
    logger.log("sampling complete")


if __name__ == "__main__":
    main()
