import torch
import torch.nn as nn

class AnomalyDetector(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # FFN layer + FC layer (no Sigmoid at the end to support logits-based BCE loss)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, f_out):
        # f_out: [Batch, Temporal, FeatureDim]
        # compute S_a = FC(FFN(F) + F)
        ffn_out = self.ffn(f_out)
        out = ffn_out + f_out
        logits = self.fc(out)
        return logits.squeeze(-1) # [Batch, Temporal]