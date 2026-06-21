import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms

# 1. Force Python to look inside your local cloned folders
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'detr'))
sys.path.append(os.path.join(current_dir, 'CLIP'))

# 2. Import directly from your local clones
import clip
from models import build_model as build_detr
import argparse

class LocalDetrClipConnector(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        # --- Initialize Local CLIP ---
        # This loads the model directly from your CLIP folder
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_dim = self.clip_model.visual.output_dim
        
        # --- Initialize Local DETR ---
        # DETR requires this argument namespace to build the architecture locally
        args = argparse.Namespace(
            lr_backbone=1e-5, masks=False, backbone='resnet50',
            dilation=False, position_embedding='sine', hidden_dim=256,
            dropout=0.1, nheads=8, dim_feedforward=2048, enc_layers=6,
            dec_layers=6, pre_norm=False, num_classes=91
        )
        # Build the model using the local detr/models/ folder
        self.detr_model, self.criterion, self.postprocessors = build_detr(args)
        
        # Load the official pretrained weights into your local architecture
        checkpoint = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
            map_location=self.device, check_hash=True)
        self.detr_model.load_state_dict(checkpoint['model'])
        self.detr_model.to(self.device)
        self.detr_dim = args.hidden_dim
        
        # --- Projection Layer (As defined in the paper) ---
        self.projection = nn.Linear(self.detr_dim, self.clip_dim)

    def forward(self, images, text_prompts):
        # 1. Process through local DETR
        # DETR expects a list of tensors or a NestedTensor
        detr_out = self.detr_model(images)
        # Extract the hidden states (last layer of encoder/decoder)
        detr_feat = detr_out['pred_logits'] # Simplified extraction for the connection
        proj_detr = self.projection(detr_feat)

        # 2. Process through local CLIP
        # Tokenize text prompts using local CLIP tokenizer
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        
        # Extract features
        with torch.no_grad():
            img_embed = self.clip_model.encode_image(images)
            txt_embed = self.clip_model.encode_text(text_tokens)

        return proj_detr, img_embed, txt_embed