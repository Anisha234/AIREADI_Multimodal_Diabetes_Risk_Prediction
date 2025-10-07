import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch

class PositionalRandomReplaceVector:
    """
    Replace x[i] with `value` with probability p_vector[i].
    - x must be a 1D tensor of length L.
    - p_vector must have length L and values in [0, 1].
    """
    def __init__(self, p_vector, value=8):
        # accept list/np/tensor; store as float tensor
        self.p_vector = torch.as_tensor(p_vector, dtype=torch.float32)
        self.value = value

        # clamp once at init (also guards small numeric drift)
        self.p_vector = torch.clamp(self.p_vector, 0.0, 1.0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        if x.dim() != 1:
            raise ValueError(f"Expected a 1D tensor, got shape {tuple(x.shape)}")

        L = x.numel()
        if self.p_vector.numel() != L:
            raise ValueError(
                f"p_vector length {self.p_vector.numel()} != tensor length {L}"
            )

        # move probs to same device as x
        p = self.p_vector.to(device=x.device)

        # sample independent Bernoulli per position
        mask = torch.rand(L, device=x.device) < p

        out = x.clone()
        out[mask] = out.new_tensor(self.value)  # cast to x.dtype/device
        return out

    def __repr__(self):
        return (f"{self.__class__.__name__}(len={self.p_vector.numel()}, "
                f"value={self.value})")


class ClinicalDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, drop_transform=None, transform=None):
        self.data = torch.tensor(dataframe.to_numpy(), dtype=torch.float32)
        self.drop_transform = drop_transform
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        original = self.data[idx]            # 1D tensor (features,)
        masked = (self.drop_transform(original)
                  if self.drop_transform is not None
                  else original.clone())

        return original, masked
