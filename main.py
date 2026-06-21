import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import AADCDataset
from model import AADCNet
from loss import AADCLoss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = AADCDataset(root_dir='.', transform=transform, frames_per_sec=30, num_frames=16)
    
    # We will use batch_size 1 since videos can be memory intensive
    # but we can try 2 if memory permits
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    text_prompts = train_dataset.get_text_prompts()
    print(f"Detected categories: {train_dataset.categories}")
    print(f"Text Prompts: {text_prompts}")
    
    model = AADCNet(clip_dim=512, detr_dim=256, mtn_dim=256).to(device)
    criterion = AADCLoss(k=5, gamma=1.0, lambda_nce=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            anomaly_scores, category_logits = model(videos, text_prompts)
            
            # gt_anomaly: 1.0 if abnormal, 0.0 if normal
            gt_anomaly = (labels > 0).float()
            gt_category = labels.long()
            
            loss = criterion(anomaly_scores, category_logits, gt_anomaly, gt_category)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss/len(train_loader)}")

if __name__ == "__main__":
    main()