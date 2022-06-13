import wandb
from train import train

sweep_config = {
  "entity": "sanjeed1722",
  "project": "cbisddsm",
  "method": "bayes",
  "metric": {
    "name": "test_dice",
    "goal": "maximize"
  },
  "parameters": {
    "base_lr": {
      "values": [0.02, 0.01, 0.0015, 0.001, 0.0001, 0.00001]
      # "distribution": "uniform",
      # "max": 0.1,
      # "min": 0.000001
    },
    "optimizer": {
      "values": ["adam", "sgd"]
    },
    "dice_ce_split": {
      "values": [0, 0.5, 1.0, 2.0, 2.5]
    }
  },
  "early_terminate": {
    "type": "hyperband",
    "max_iter": 30,
    "s": 1,
    "eta": 3
  },
}

# sweep_id = wandb.sweep(sweep_config, project="cbisddsm", entity="sanjeed1722")
print("sweep_id", 'r07jlmeh')

wandb.agent('sanjeed1722/cbisddsm/r07jlmeh', train)


