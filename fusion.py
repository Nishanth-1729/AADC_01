import torch
import torch.nn as nn

class DetrClipFusion(nn.Module):
    def __init__(self, clip_dim, detr_dim):
        super().__init__()
        self.f_fc = nn.Linear(clip_dim, detr_dim)

    def forward(self, f_vis, f_detr):
        projected_clip = self.f_fc(f_vis)
        x_c = projected_clip + f_detr
        return x_c