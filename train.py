import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import AADCDataset
from model import AADCNet
from loss import AADCLoss

def train_aadc_net(data_dir, num_epochs=10, batch_size=1, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = AADCDataset(root_dir=data_dir, transform=transform, frames_per_sec=30)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = AADCNet().to(device)
    criterion = AADCLoss(k=5, gamma=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    text_prompts = ["normal workout", "abnormal activity"]
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (videos, labels) in enumerate(dataloader):
            videos = videos.squeeze(0).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            anomaly_scores, category_logits = model(videos, text_prompts)
            anomaly_scores = anomaly_scores.mean(dim=-1).unsqueeze(0)
            
            loss = criterion(anomaly_scores, category_logits, labels.unsqueeze(0), labels.long())
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

if __name__ == "__main__":
    train_aadc_net("dataset_path")