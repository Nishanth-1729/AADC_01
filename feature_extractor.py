import torch
import torch.nn as nn
from transformers import DetrImageProcessor, DetrForObjectDetection, CLIPProcessor, CLIPModel
from torchvision.ops import roi_align

class DetrClipConnector(nn.Module):
    def __init__(self, detr_name='facebook/detr-resnet-50', clip_name='openai/clip-vit-base-patch32', top_k_boxes=5):
        super().__init__()
        self.detr_processor = DetrImageProcessor.from_pretrained(detr_name)
        self.detr = DetrForObjectDetection.from_pretrained(detr_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_name)
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.top_k_boxes = top_k_boxes

    def forward(self, images):
        device = next(self.detr.parameters()).device
        
        # Unnormalize images back to [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        unnormalized_images = images * std + mean
        unnormalized_images = torch.clamp(unnormalized_images, 0.0, 1.0)
        
        with torch.no_grad():
            detr_out = self.detr(pixel_values=images)
            
        probs = detr_out.logits.softmax(dim=-1)
        scores = probs[..., :-1].max(dim=-1)[0]
        
        topk_values, topk_indices = torch.topk(scores, self.top_k_boxes, dim=1)
        
        batch_size, num_queries, hidden_dim = detr_out.last_hidden_state.shape
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        detr_feat = torch.gather(detr_out.last_hidden_state, dim=1, index=topk_indices_expanded)
        
        topk_boxes = torch.gather(
            detr_out.pred_boxes, 1, 
            topk_indices.unsqueeze(-1).expand(-1, -1, 4)
        )
        
        cx, cy, bw, bh = topk_boxes.unbind(-1)
        x1 = (cx - 0.5 * bw).clamp(min=0.0, max=1.0)
        y1 = (cy - 0.5 * bh).clamp(min=0.0, max=1.0)
        x2 = (cx + 0.5 * bw).clamp(min=0.0, max=1.0)
        y2 = (cy + 0.5 * bh).clamp(min=0.0, max=1.0)
        
        h, w = unnormalized_images.shape[2], unnormalized_images.shape[3]
        
        x1_abs = x1 * w
        y1_abs = y1 * h
        x2_abs = x2 * w
        y2_abs = y2 * h
        
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, self.top_k_boxes)
        
        rois = torch.stack([
            batch_indices.float(), x1_abs, y1_abs, x2_abs, y2_abs
        ], dim=-1).view(-1, 5)
        
        invalid_boxes = (rois[:, 3] <= rois[:, 1]) | (rois[:, 4] <= rois[:, 2])
        rois[invalid_boxes, 3] = rois[invalid_boxes, 1] + 1e-3
        rois[invalid_boxes, 4] = rois[invalid_boxes, 2] + 1e-3
        
        cropped_patches = roi_align(unnormalized_images, rois, output_size=(224, 224), spatial_scale=1.0)
        
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        clip_in_img_tensor = (cropped_patches - clip_mean) / clip_std
        
        with torch.no_grad():
            img_embed = self.clip.get_image_features(pixel_values=clip_in_img_tensor)
            if hasattr(img_embed, 'pooler_output'):
                img_embed = img_embed.pooler_output
            elif hasattr(img_embed, 'image_embeds'):
                img_embed = img_embed.image_embeds
                
        img_embed = img_embed.view(batch_size, self.top_k_boxes, -1)
        
        return detr_feat, img_embed