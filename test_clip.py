import torch
from transformers import CLIPModel, CLIPProcessor
try:
    clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    out = clip.get_image_features(pixel_values=torch.randn(1,3,224,224))
    print(type(out))
    if hasattr(out, 'keys'):
        print(out.keys())
    if hasattr(out, 'pooler_output'):
        print("Has pooler_output")
    if hasattr(out, 'image_embeds'):
        print("Has image_embeds")
except Exception as e:
    print(e)
