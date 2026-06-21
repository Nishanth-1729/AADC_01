import torch
import torch.nn as nn

class PDCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dilated1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.dilated2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.dilated4 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)

    def forward(self, x):
        d1 = self.dilated1(x)
        d2 = self.dilated2(x)
        d4 = self.dilated4(x)
        return d1, d2, d4

class TSABlock(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        reduced_dim = feature_dim // 4
        self.conv1x1_reduce = nn.Conv1d(feature_dim, reduced_dim, kernel_size=1)
        self.conv1x1_q = nn.Conv1d(reduced_dim, reduced_dim, kernel_size=1)
        self.conv1x1_k = nn.Conv1d(reduced_dim, reduced_dim, kernel_size=1)
        self.conv1x1_v = nn.Conv1d(reduced_dim, reduced_dim, kernel_size=1)
        self.conv1x1_out = nn.Conv1d(reduced_dim, reduced_dim, kernel_size=1)

    def forward(self, x):
        x_c = self.conv1x1_reduce(x)
        q = self.conv1x1_q(x_c)
        k = self.conv1x1_k(x_c)
        v = self.conv1x1_v(x_c)
        
        attention_map = torch.bmm(q.transpose(1, 2), k)
        attention_map = torch.softmax(attention_map, dim=-1)
        
        out = torch.bmm(v, attention_map.transpose(1, 2))
        x_c4 = self.conv1x1_out(out)
        
        x_tsa = x_c4 + x_c
        return x_tsa

class MTN(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.pdc = PDCBlock(feature_dim, feature_dim)
        self.tsa = TSABlock(feature_dim)
        reduced_dim = feature_dim // 4
        self.combine = nn.Conv1d(feature_dim * 3 + reduced_dim, feature_dim, kernel_size=1)

    def forward(self, x):
        d1, d2, d4 = self.pdc(x)
        x_tsa = self.tsa(x)
        
        concat_features = torch.cat([d1, d2, d4, x_tsa], dim=1)
        combined = self.combine(concat_features)
        
        f_out = combined + x
        return f_out