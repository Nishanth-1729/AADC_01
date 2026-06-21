import torch
import torch.nn as nn
from feature_extractor import DetrClipConnector
from fusion import DetrClipFusion
from mtn import MTN
from anomaly_detector import AnomalyDetector

class AADCNet(nn.Module):
    def __init__(self, clip_dim=512, detr_dim=256, mtn_dim=256):
        super().__init__()
        self.feature_extractor = DetrClipConnector()
        # Ensure we don't update DETR/CLIP parameters for faster training unless required
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.fusion = DetrClipFusion(clip_dim=clip_dim, detr_dim=detr_dim)
        self.mtn = MTN(feature_dim=mtn_dim)
        self.anomaly_predictor = AnomalyDetector(feature_dim=mtn_dim)
        
    def forward(self, videos, text_prompts):
        """
        videos: [Batch, Temporal, C, H, W]
        text_prompts: list of strings (categories)
        """
        batch_size, num_frames, c, h, w = videos.shape
        videos_flat = videos.view(batch_size * num_frames, c, h, w)
        
        # Extract features
        # proj_detr: [Batch*T, num_queries, detr_dim]
        # img_embed: [Batch*T, clip_dim]
        # txt_embed: [num_classes, clip_dim]
        proj_detr, img_embed, txt_embed = self.feature_extractor(videos_flat, text_prompts)
        
        # We need to pool proj_detr across queries to match image embedding, or broadcast img_embed
        # Let's mean-pool proj_detr across queries: [Batch*T, detr_dim]
        proj_detr_pooled = proj_detr.mean(dim=1)
        
        # Fusion
        # x_c: [Batch*T, detr_dim]
        x_c = self.fusion(img_embed, proj_detr_pooled)
        
        # Reshape for MTN: [Batch, Temporal, detr_dim]
        # MTN expects [Batch, Feature, Temporal] ? Wait, let's check MTN
        # mtn.py: nn.Conv1d expects [Batch, in_channels, sequence_length]
        x_c_reshaped = x_c.view(batch_size, num_frames, -1).transpose(1, 2)
        
        # MTN
        # f_out: [Batch, detr_dim, Temporal]
        f_out = self.mtn(x_c_reshaped)
        
        # Transpose back for anomaly predictor: [Batch, Temporal, detr_dim]
        f_out_transposed = f_out.transpose(1, 2)
        
        # Anomaly Predictor
        # anomaly_scores: [Batch, Temporal]
        anomaly_scores = self.anomaly_predictor(f_out_transposed)
        
        # For category logits, we compute similarity between fused visual features and textual embeddings.
        # video_level_features: [Batch, clip_dim]
        # Wait, the paper: "visual CLIP features are fused with anomaly confidence... then leverages text encoder"
        # We can just average the visual features weighted by anomaly score to get video level features.
        weighted_img_embed = (img_embed.view(batch_size, num_frames, -1) * anomaly_scores.unsqueeze(-1)).mean(dim=1)
        # txt_embed: [num_classes, clip_dim]
        # Calculate logits: [Batch, num_classes]
        # normalize
        weighted_img_embed = weighted_img_embed / weighted_img_embed.norm(dim=-1, keepdim=True)
        txt_embed = txt_embed / txt_embed.norm(dim=-1, keepdim=True)
        category_logits = torch.matmul(weighted_img_embed, txt_embed.t())
        
        return anomaly_scores, category_logits