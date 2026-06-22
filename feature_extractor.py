import torch
import torch.nn as nn
from torchvision.transforms import functional as F_vision
from transformers import DetrImageProcessor, DetrModel, CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

class DetrClipConnector(nn.Module):
    def __init__(self, detr_name='facebook/detr-resnet-50', clip_name='openai/clip-vit-base-patch32', top_k_boxes=5):
        super().__init__()
        self.detr_processor = DetrImageProcessor.from_pretrained(detr_name)
        self.detr = DetrModel.from_pretrained(detr_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_name)
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.top_k_boxes = top_k_boxes

    def forward(self, images, text_prompts):
        device = next(self.detr.parameters()).device
        
        pil_images = []
        for img_tensor in images:
            img_np = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))
        
        detr_in = self.detr_processor(images=pil_images, return_tensors="pt")
        detr_in = {k: v.to(device) for k, v in detr_in.items()}
        
        with torch.no_grad():
            detr_out = self.detr(**detr_in)
            
        detr_feat = detr_out.last_hidden_state 
        
        batch_patches = []
        
        for i in range(len(images)):
            boxes = detr_out.encoder_last_hidden_state[i, :self.top_k_boxes, :]
            img = images[i]
            
            h, w = img.shape[1], img.shape[2]
            
            for j in range(self.top_k_boxes):
                box_feat = boxes[j]
                
                cx = (box_feat[0].item() % 1.0) * w
                cy = (box_feat[1].item() % 1.0) * h
                bw = (box_feat[2].item() % 1.0) * w
                bh = (box_feat[3].item() % 1.0) * h
                
                x1 = max(0, int(cx - bw / 2))
                y1 = max(0, int(cy - bh / 2))
                x2 = min(w, int(cx + bw / 2))
                y2 = min(h, int(cy + bh / 2))
                
                if x2 <= x1 or y2 <= y1:
                    patch = img
                else:
                    patch = F_vision.crop(img, y1, x1, y2 - y1, x2 - x1)
                    
                patch_np = (patch.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                batch_patches.append(Image.fromarray(patch_np))
                
        clip_in_img = self.clip_processor(images=batch_patches, return_tensors="pt")
        clip_in_img = {k: v.to(device) for k, v in clip_in_img.items()}
        
        with torch.no_grad():
            img_embed = self.clip.get_image_features(**clip_in_img)
            
        img_embed = img_embed.view(len(images), self.top_k_boxes, -1).mean(dim=1)
        
        clip_in_txt = self.clip_processor(text=text_prompts, return_tensors="pt", padding=True)
        clip_in_txt = {k: v.to(device) for k, v in clip_in_txt.items()}
        
        with torch.no_grad():
            txt_embed = self.clip.get_text_features(**clip_in_txt)

        return detr_feat, img_embed, txt_embed