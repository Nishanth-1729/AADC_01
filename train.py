import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import AADCDataset
from model import AADCNet
from loss import AADCLoss

def train_aadc_net(data_dir, num_epochs=10, batch_size=2, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = AADCDataset(root_dir=data_dir, transform=transform, num_frames=16)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    text_prompts = dataset.get_text_prompts()
    num_classes = len(dataset.categories)
    
    model = AADCNet(num_classes=num_classes).to(device)
    criterion = AADCLoss(k=5, gamma=1.0)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (videos, labels) in enumerate(dataloader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            anomaly_scores, category_logits = model(videos, text_prompts)
            
            gt_anomaly = (labels > 0).float()
            gt_category = labels.long()
            
            loss = criterion(anomaly_scores, category_logits, gt_anomaly, gt_category)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

if __name__ == "__main__":
    train_aadc_net("dataset_path")