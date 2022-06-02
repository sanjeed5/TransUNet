import wandb
from train import main
import os

sweep_config = {
  "method": "random",
  "metric": {
    "name": "val_loss",
    "goal": "minimize"
  },
  "parameters": {
    "base_lr": {
      "distribution": "uniform",
      "max": 0.1,
      "min": 0
    },
    "optimizer": {
      "values": ["adam", "sgd"]
    },
    "dice_ce_split": {
      "distribution": "q_uniform",
      "max": 1,
      "min": 0,
      "q": 0.1
    }
  },
  "early_terminate": {
    "type": "hyperband",
    "max_iter": 30,
    "s": 1,
    "eta": 3
  },
}

sweep_id = wandb.sweep(sweep_config, project="cbisddsm", entity="sanjeed1722")
print("sweep_id", sweep_id)

wandb.agent(sweep_id, main, count=5)


