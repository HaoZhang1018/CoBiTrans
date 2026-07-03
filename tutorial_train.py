from share import *

import os
import math
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Sampler
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint


class ConsecutiveDistributedSampler(Sampler):

    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas,
        rank,
        drop_last=True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last

        # 一个全局 step 一共使用的样本数
        self.global_batch_size = batch_size * num_replicas

        if self.drop_last:
            # 丢弃不能构成完整全局 batch 的尾部样本
            self.num_global_batches = (
                len(self.dataset) // self.global_batch_size
            )
        else:
            # 补齐最后一个全局 batch
            self.num_global_batches = math.ceil(
                len(self.dataset) / self.global_batch_size
            )

        # 每个 GPU 每轮读取的样本数
        self.num_samples = self.num_global_batches * self.batch_size

        # 所有 GPU 每轮总共读取的样本数
        self.total_size = (
            self.num_global_batches * self.global_batch_size
        )

    def __iter__(self):
        # 严格按照 JSON 对应的数据索引顺序
        indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # 当数据量不能整除全局 batch 时，从头部复制样本进行补齐
            padding_size = self.total_size - len(indices)

            if padding_size > 0:
                repeat_times = math.ceil(
                    padding_size / len(indices)
                )
                padding_indices = (
                    indices * repeat_times
                )[:padding_size]

                indices += padding_indices
        else:
            # 丢弃最后不足一个完整全局 batch 的样本
            indices = indices[:self.total_size]

        local_indices = []

        # 每次取一个全局 batch
        for global_start in range(
            0,
            self.total_size,
            self.global_batch_size,
        ):
            # 当前 rank 在全局 batch 中的连续区间
            local_start = (
                global_start
                + self.rank * self.batch_size
            )
            local_end = local_start + self.batch_size

            local_indices.extend(
                indices[local_start:local_end]
            )

        return iter(local_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        # 不进行 shuffle，因此不需要根据 epoch 攱变随机种子
        pass


# =========================================================
# Configs
# =========================================================

resume_path = "./models/control_sd21_ini.ckpt"

batch_size = 2
logger_freq = 500
learning_rate = 5e-5

sd_locked = True
only_mid_control = False

gpu_ids = [1, 2, 3]
world_size = len(gpu_ids)


# =========================================================
# 获取当前 DDP 进程的 local rank
# =========================================================
local_rank = int(os.environ.get("LOCAL_RANK", 0))

print(
    f"LOCAL_RANK={local_rank}, "
    f"physical GPU={gpu_ids[local_rank]}"
)


# =========================================================
# Model
# =========================================================

model = create_model("./models/cldm_v21.yaml").cpu()

model.load_state_dict(
    load_state_dict(
        resume_path,
        location="cpu",
    )
)

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# =========================================================
# Dataset and sampler
# =========================================================

dataset = MyDataset()

sampler = ConsecutiveDistributedSampler(
    dataset=dataset,
    batch_size=batch_size,
    num_replicas=world_size,
    rank=local_rank,
    drop_last=True,
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    shuffle=False,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)


# =========================================================
# Logger and checkpoint
# =========================================================

logger = ImageLogger(
    batch_frequency=logger_freq
)

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",
    filename="model-{epoch:02d}",
    save_top_k=-1,
    every_n_epochs=1,
)


# =========================================================
# Trainer
# =========================================================

trainer = pl.Trainer(
    gpus=gpu_ids,
    precision=32,
    strategy="ddp",
    max_steps=600000,
    callbacks=[
        logger,
        checkpoint_callback,
    ],
    replace_sampler_ddp=False,
)


if __name__ == "__main__":
    trainer.fit(
        model,
        dataloader,
    )