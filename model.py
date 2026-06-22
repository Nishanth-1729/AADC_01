import torch
import torch.nn as nn
from feature_extractor import DetrClipConnector
from fusion import DetrClipFusion
from mtn import MTN
from anomaly_detector import AnomalyDetector

class LearnablePrompt(nn.Module):
    def __init__(self, clip_dim, num_classes, num_ctx=4):
        super().__init__()
        self.num_ctx = num_ctx
        self.ctx = nn.Parameter(torch.empty(num_classes, num_ctx, clip_dim))
        nn.init.normal_(self.ctx, std=0.02)

    def forward(self, text_embeddings):
        ctx_expanded = self.ctx.mean(dim=1) 
        enhanced_text = text_embeddings + ctx_expanded
        return enhanced_text

class AADCNet(nn.Module):
    def __init__(self, num_classes, clip_dim=512, detr_dim=256, mtn_dim=256):
        super().__init__()
        self.feature_extractor = DetrClipConnector()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.fusion = DetrClipFusion(clip_dim=clip_dim, detr_dim=detr_dim)
        self.mtn = MTN(feature_dim=mtn_dim)
        self.anomaly_predictor = AnomalyDetector(feature_dim=mtn_dim)
        self.prompt_learner = LearnablePrompt(clip_dim=clip_dim, num_classes=num_classes)

    def train(self, mode=True):
        super().train(mode)
        self.feature_extractor.eval()
        return self
        
    def forward(self, videos, text_prompts):
        batch_size, num_frames, c, h, w = videos.shape
        videos_flat = videos.view(batch_size * num_frames, c, h, w)
        
        proj_detr, img_embed, txt_embed = self.feature_extractor(videos_flat, text_prompts)
        
        txt_embed = self.prompt_learner(txt_embed)
        
        proj_detr_pooled = proj_detr.mean(dim=1)
        
        x_c = self.fusion(img_embed, proj_detr_pooled)
        
        x_c_reshaped = x_c.view(batch_size, num_frames, -1).transpose(1, 2)
        
        f_out = self.mtn(x_c_reshaped)
        
        f_out_transposed = f_out.transpose(1, 2)
        
        anomaly_scores = self.anomaly_predictor(f_out_transposed)
        
        weighted_img_embed = (img_embed.view(batch_size, num_frames, -1) * anomaly_scores.unsqueeze(-1)).mean(dim=1)
        
        weighted_img_embed = weighted_img_embed / weighted_img_embed.norm(dim=-1, keepdim=True)
        txt_embed = txt_embed / txt_embed.norm(dim=-1, keepdim=True)
        
        category_logits = torch.matmul(weighted_img_embed, txt_embed.t())
        
        return anomaly_scores, category_logits