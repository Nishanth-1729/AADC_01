import torch
import torch.nn as nn

class AADCLoss(nn.Module):
    def __init__(self, k=5, gamma=1.0, lambda_nce=1.0):
        super().__init__()
        self.k = k
        self.gamma = gamma
        self.lambda_nce = lambda_nce
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()

    def get_top_k_scores(self, scores):
        if scores.size(1) < self.k:
            return torch.mean(scores, dim=1)
        topk_scores, _ = torch.topk(scores, self.k, dim=1)
        return torch.mean(topk_scores, dim=1)

    def forward(self, anomaly_scores, category_logits, gt_anomaly, gt_category):
        top_k_anomaly = self.get_top_k_scores(anomaly_scores)
        loss_bce = self.bce(top_k_anomaly, gt_anomaly)
        
        scaled_logits = category_logits * self.gamma
        loss_nce = self.ce(scaled_logits, gt_category)
        
        total_loss = loss_bce + self.lambda_nce * loss_nce
        return total_loss