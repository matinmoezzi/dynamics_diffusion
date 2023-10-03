import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import pickle as pkl
from sacsvg.train import Workspace


OmegaConf.register_new_resolver("split", lambda x: x.split(".")[-1])


@hydra.main(
    version_base=None, config_path="../config/sacsvg_config", config_name="train_sacsvg"
)
def main(cfg: DictConfig):
    log_dir = Path(HydraConfig.get().run.dir).resolve()

    fname = os.getcwd() + "/latest.pkl"
    if os.path.exists(fname):
        print(f"Resuming fom {fname}")
        with open(fname, "rb") as f:
            workspace = pkl.load(f)
    else:
        workspace = Workspace(cfg, log_dir=log_dir)

    workspace.run()


if __name__ == "__main__":
    main()
