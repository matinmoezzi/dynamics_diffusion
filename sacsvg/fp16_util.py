from dynamics_diffusion.fp16_util import *


class MixedPrecisionTrainer(MixedPrecisionTrainer):
    def optimize(self, opt: th.optim.Optimizer, step):
        if self.use_fp16:
            return self._optimize_fp16(opt, step)
        else:
            return self._optimize_normal(opt, step)

    def _optimize_fp16(self, opt: th.optim.Optimizer, step):
        logger.logkv_mean("train_diffusion/lg_loss_scale", self.lg_loss_scale, step)
        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2**self.lg_loss_scale)
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            zero_master_grads(self.master_params)
            return False

        logger.logkv_mean("train_diffusion/grad_norm", grad_norm, step)
        logger.logkv_mean("train_diffusion/param_norm", param_norm, step)

        for p in self.master_params:
            p.grad.mul_(1.0 / (2**self.lg_loss_scale))
        opt.step()
        zero_master_grads(self.master_params)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt: th.optim.Optimizer, step):
        grad_norm, param_norm = self._compute_norms()
        logger.logkv_mean("train_diffusion/grad_norm", grad_norm, step)
        logger.logkv_mean("train_diffusion/param_norm", param_norm, step)
        opt.step()
        return True
