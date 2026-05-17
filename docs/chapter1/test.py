import wandb
wandb.init(
    project="cs336-a5-sft-v2",          # 项目名（必选）
    name="wanda_sft",        # 可读 run 名称
    config={
        "model": "Qwen2.5-Math-1.5B",
        "dataset_tag": "raw", # raw, sf, grpo
        "batch_size": 64,
        "max_examples": "1000",
        "seed": 2026,
        "learning_rate": 2e-5,
    }
)
import time
import random
for i in range(1000):
    time.sleep(0.5)
    wandb.log({
        "train/step": i,
        "train/loss": 1 - 0.001 * i + random.random()
    })