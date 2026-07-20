import os
import json
import cv2
import torch
import einops
import numpy as np

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

# =====================
# 配置
# =====================
config_path = "./models/cldm_v21.yaml"
ckpt_path = "./checkpoints/last.ckpt"
json_path = "./dataset/test.json"
data_root = "./dataset"
save_dir = "./results"

ddim_steps = 50
ddim_eta = 0.0
guidance_scale = 3.0
device = torch.device("cuda")

os.makedirs(save_dir, exist_ok=True)

# =====================
# 随机种子
# =====================
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# =====================
# 模型初始化
# =====================
model = create_model(config_path).cpu()
model.load_state_dict(load_state_dict(ckpt_path, location=device))
model = model.to(device).eval()

ddim_sampler = DDIMSampler(model)

# =====================
# 读取 JSON
# =====================
with open(json_path, "r") as f:
    meta = json.load(f)["data"]

# =====================
# 遍历 JSON
# =====================
for idx, item in enumerate(meta):
    print(f"[{idx + 1}/{len(meta)}] Processing {item['source']}")

    src_path = os.path.join(data_root, item["source"])
    prompt = item["prompt"]
    negative_prompt = ""

    # ---------- 读输入图 ----------
    input_image = cv2.imread(src_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    img = input_image.astype(np.float32) / 255.0
    H, W, _ = img.shape

    control = torch.from_numpy(img).float().unsqueeze(0).to(device)
    control = einops.rearrange(control, 'b h w c -> b c h w')
    control = control.contiguous()

    # ---------- 条件 ----------
    cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning([prompt])]
    }

    un_cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning([negative_prompt])]
    }

    shape = (4, H // 8, W // 8)

    # ---------- 推理 ----------
    with torch.no_grad():
        samples, _ = ddim_sampler.sample(
            S=ddim_steps,
            batch_size=1,
            shape=shape,
            conditioning=cond,
            eta=ddim_eta,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=un_cond,
            verbose=False
        )

        x_samples = model.decode_first_stage(samples)
        x_samples = einops.rearrange(x_samples, 'b c h w -> b h w c')
        x_samples = x_samples
        x_samples = (x_samples * 127.5 + 127.5).clamp(0, 255)
        result = x_samples[0].cpu().numpy().astype(np.uint8)

    output_vis = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    save_rel_path = item["target"].replace(".jpg", ".png")
    save_path = os.path.join(save_dir, save_rel_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, output_vis)
    print(f"    Saved -> {save_path}")

print(" All done.")
