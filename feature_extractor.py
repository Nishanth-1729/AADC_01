import torch
import torch.nn as nn
from transformers import DetrImageProcessor, DetrModel, CLIPProcessor, CLIPModel

class DetrClipConnector(nn.Module):
    def __init__(self, detr_name='facebook/detr-resnet-50', clip_name='openai/clip-vit-base-patch32'):
        super().__init__()
        self.detr_processor = DetrImageProcessor.from_pretrained(detr_name)
        self.detr = DetrModel.from_pretrained(detr_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_name)
        self.clip = CLIPModel.from_pretrained(clip_name)
        # We don't project detr features here, fusion.py projects CLIP to match DETR.

    def forward(self, images, text_prompts):
        detr_in = self.detr_processor(images=images, return_tensors="pt")
        detr_in = {k: v.to(self.detr.device) for k, v in detr_in.items()}
        detr_out = self.detr(**detr_in)
        detr_feat = detr_out.last_hidden_state

        clip_in = self.clip_processor(text=text_prompts, images=images, return_tensors="pt", padding=True)
        clip_in = {k: v.to(self.clip.device) for k, v in clip_in.items()}
        clip_out = self.clip(**clip_in)
        img_embed = clip_out.image_embeds
        txt_embed = clip_out.text_embeds

        return detr_feat, img_embed, txt_embed