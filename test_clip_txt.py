import torch
from transformers import CLIPModel, CLIPProcessor
try:
    clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    
    inputs = proc(text=["a photo of a cat"], return_tensors="pt")
    out_txt = clip.get_text_features(**inputs)
    print("TXT type:", type(out_txt))
    if hasattr(out_txt, 'keys'):
        print("TXT keys:", out_txt.keys())
        
    out_img = clip.get_image_features(pixel_values=torch.randn(1,3,224,224))
    print("IMG type:", type(out_img))
    
    if hasattr(out_img, 'pooler_output'):
        print("IMG pooler shape:", out_img.pooler_output.shape)
        if hasattr(clip, 'visual_projection'):
            proj = clip.visual_projection(out_img.pooler_output)
            print("Projected IMG shape:", proj.shape)
            
except Exception as e:
    print(e)
