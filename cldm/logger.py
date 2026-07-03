import os
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency=2000,
        max_images=4,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None
    ):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    # =========================
    # 只在 rank0 写文件
    # =========================
    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        os.makedirs(root, exist_ok=True)

        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            if self.rescale:
                grid = (grid + 1.0) / 2.0  # [-1,1] -> [0,1]

            grid = grid.permute(1, 2, 0).cpu().numpy()
            grid = (grid * 255).astype(np.uint8)

            filename = f"{k}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.png"
            path = os.path.join(root, filename)

            Image.fromarray(grid).save(path)

    # =========================
    # 只让 rank0 进入采样逻辑
    # =========================
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if not pl_module.trainer.is_global_zero:
            return

        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step

        if not (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            return

        was_training = pl_module.training
        if was_training:
            pl_module.eval()

        with torch.no_grad():
            images = pl_module.log_images(
                batch,
                split=split,
                **self.log_images_kwargs
            )

        for k in images:
            N = min(images[k].shape[0], self.max_images)
            images[k] = images[k][:N]

            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

        self.log_local(
            pl_module.logger.save_dir,
            split,
            images,
            pl_module.global_step,
            pl_module.current_epoch,
            batch_idx
        )

        if was_training:
            pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    # =========================
    # Lightning 回调入口
    # =========================
    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0
    ):
        if self.disabled:
            return

        self.log_img(pl_module, batch, batch_idx, split="train")
