from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class EarlyStopping:
    patience: int = 3
    min_delta: float = 0.0
    save_path: str = "checkpoints/best_model.pt"
    mode: str = "min"

    def __post_init__(self):
        self.best = None
        self.bad_epochs = 0
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

    def step(self, value: float, model: torch.nn.Module) -> bool:
        improved = False

        if self.best is None:
            improved = True
        else:
            if self.mode == "min":
                improved = (self.best - value) > self.min_delta
            else:
                improved = (value - self.best) > self.min_delta

        if improved:
            self.best = value
            self.bad_epochs = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.patience