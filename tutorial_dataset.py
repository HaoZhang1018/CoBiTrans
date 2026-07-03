import json
import cv2
import numpy as np
import random

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        # 读取 JSON 数据
        with open('./dataset/train.json', 'r') as f:
            self.data = json.load(f)["data"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        # prompt = "None"

        source = cv2.imread('dataset_source/' + source_filename)
        
        if random.random() < 0.45:
            if prompt:
                prompt = prompt.split('.', 1)[0].strip()
            target = cv2.imread('dataset_enhance/' + target_filename)
        else:
            target = cv2.imread('dataset_control/' + target_filename)


        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


