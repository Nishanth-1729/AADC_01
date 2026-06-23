import torch
import torch.nn as nn
from feature_extractor import DetrClipConnector
from fusion import DetrClipFusion
from mtn import MTN
from anomaly_detector import AnomalyDetector

class LearnablePrompt(nn.Module):
    def __init__(self, clip_model, clip_processor, num_classes, num_ctx=4):
        super().__init__()
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.num_ctx = num_ctx
        
        hidden_size = clip_model.text_model.config.hidden_size
        self.ctx = nn.Parameter(torch.empty(num_classes, num_ctx, hidden_size))
        nn.init.normal_(self.ctx, std=0.02)

    def forward(self, text_labels):
        device = self.ctx.device
        inputs = self.clip_processor(text=text_labels, return_tensors="pt", padding=True).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        with torch.no_grad():
            token_embeds = self.clip_model.text_model.embeddings.token_embedding(input_ids)
            
        bos_embeds = token_embeds[:, 0:1, :]
        rest_embeds = token_embeds[:, 1:, :]
        inputs_embeds = torch.cat([bos_embeds, self.ctx, rest_embeds], dim=1)
        
        ctx_mask = torch.ones(len(text_labels), self.num_ctx, dtype=attention_mask.dtype, device=device)
        extended_attention_mask = torch.cat([attention_mask[:, 0:1], ctx_mask, attention_mask[:, 1:]], dim=1)
        
        class PatchedEmbeddings(nn.Module):
            def __init__(self, orig_emb, custom_embeds):
                super().__init__()
                self.orig_emb = orig_emb
                self.custom_embeds = custom_embeds
            def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
                return self.orig_emb(inputs_embeds=self.custom_embeds, position_ids=position_ids)
                
        orig_emb = self.clip_model.text_model.embeddings
        self.clip_model.text_model.embeddings = PatchedEmbeddings(orig_emb, inputs_embeds)
        
        # Pass dummy input_ids to bypass the hardcoded ValueError check in transformers
        dummy_input_ids = torch.zeros((inputs_embeds.shape[0], inputs_embeds.shape[1]), dtype=torch.long, device=device)
        
        text_outputs = self.clip_model.text_model(
            input_ids=dummy_input_ids,
            attention_mask=extended_attention_mask,
            output_hidden_states=False
        )
        
        # Restore the original embeddings
        self.clip_model.text_model.embeddings = orig_emb
        
        last_hidden_state = text_outputs.last_hidden_state
        eos_indices = input_ids.argmax(dim=-1) + self.num_ctx
        
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=device),
            eos_indices
        ]
        
        text_features = self.clip_model.text_projection(pooled_output)
        return text_features


class AADCNet(nn.Module):
    def __init__(self, num_classes, clip_dim=512, detr_dim=256, mtn_dim=256):
        super().__init__()
        self.feature_extractor = DetrClipConnector()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.fusion = DetrClipFusion(clip_dim=clip_dim, detr_dim=detr_dim)
        self.mtn = MTN(feature_dim=mtn_dim)
        self.anomaly_predictor = AnomalyDetector(feature_dim=mtn_dim)
        
        self.prompt_learner = LearnablePrompt(
            clip_model=self.feature_extractor.clip,
            clip_processor=self.feature_extractor.clip_processor,
            num_classes=num_classes,
            num_ctx=4
        )

    def train(self, mode=True):
        super().train(mode)
        self.feature_extractor.eval()
        return self
        
    def forward(self, videos, text_prompts):
        batch_size, num_frames, c, h, w = videos.shape
        videos_flat = videos.view(batch_size * num_frames, c, h, w)
        
        proj_detr, img_embed = self.feature_extractor(videos_flat)
        txt_embed = self.prompt_learner(text_prompts)
        
        x_c = self.fusion(img_embed, proj_detr)
        x_c_pooled = x_c.mean(dim=1)
        x_c_reshaped = x_c_pooled.view(batch_size, num_frames, -1).transpose(1, 2)
        
        f_out = self.mtn(x_c_reshaped)
        f_out_transposed = f_out.transpose(1, 2)
        
        anomaly_logits = self.anomaly_predictor(f_out_transposed)
        anomaly_probs = torch.sigmoid(anomaly_logits)
        
        img_embed_avg = img_embed.mean(dim=1)
        weighted_img_embed = (img_embed_avg.view(batch_size, num_frames, -1) * anomaly_probs.unsqueeze(-1)).mean(dim=1)
        
        weighted_img_embed = weighted_img_embed / (weighted_img_embed.norm(dim=-1, keepdim=True) + 1e-8)
        txt_embed = txt_embed / (txt_embed.norm(dim=-1, keepdim=True) + 1e-8)
        
        category_logits = torch.matmul(weighted_img_embed, txt_embed.t())
        
        return anomaly_logits, category_logits