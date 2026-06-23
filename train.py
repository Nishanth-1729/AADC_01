import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import AADCDataset
from model import AADCNet
from loss import AADCLoss

def train_aadc_net(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Please check your GPU setup in WSL.")
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = AADCDataset(root_dir=args.data_dir, transform=transform, num_frames=args.num_frames)
    
    from splits import stratified_video_split, print_split_summary
    from evaluate import evaluate_video_level
    
    train_ds, val_ds, test_ds = stratified_video_split(dataset)
    print_split_summary(dataset, train_ds, val_ds, test_ds)
    
    # Expose batch size as command line argument
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    text_prompts = dataset.get_text_prompts()
    print(f"Detected categories: {dataset.categories}")
    print(f"Text Prompts: {text_prompts}")
    
    num_classes = len(dataset.categories)
    
    model = AADCNet(num_classes=num_classes).to(device)
    criterion = AADCLoss(k=args.k, gamma=args.gamma, lambda_nce=args.lambda_nce)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    start_epoch = 0
    if args.resume_from:
        if os.path.exists(args.resume_from):
            checkpoint = torch.load(args.resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from checkpoint: {args.resume_from} (starting at epoch {start_epoch+1})")
        else:
            print(f"Checkpoint path not found: {args.resume_from}. Starting from scratch.")
            
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            anomaly_logits, category_logits = model(videos, text_prompts)
            
            gt_anomaly = (labels >= dataset.num_normal_classes).float()
            gt_category = labels.long()
            
            loss = criterion(anomaly_logits, category_logits, gt_anomaly, gt_category)
            
            if torch.isnan(loss):
                raise ValueError(f"Loss became NaN at epoch {epoch+1}, batch {batch_idx}")
                
            loss.backward()
            
            # Gradient clipping to stabilize learning
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
        epoch_loss = total_loss / len(train_loader)
        print(f"--> Epoch {epoch+1}/{args.epochs} Complete. Average Loss: {epoch_loss:.4f}")
        
        # Validation
        print("Running validation...")
        try:
            val_metrics = evaluate_video_level(model, val_loader, device, dataset.num_normal_classes)
            print(f"--> Validation AUC: {val_metrics['AUC']:.4f} | AP: {val_metrics['AP']:.4f} (n={val_metrics['n_videos']})")
        except ValueError as e:
            print(f"Validation skipped: {e}")
        
        # Save checkpoint at the end of each epoch
        checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'categories': dataset.categories,
            'normal_categories': dataset.normal_categories,
            'abnormal_categories': dataset.abnormal_categories,
            'loss': epoch_loss
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description="AADC-Net Training Pipeline")
    parser.add_argument('--data_dir', type=str, default='/mnt/d/Video Analytics/UCF Merged', help="Path to normal and abnormal folders")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--resume_from', type=str, default=None, help="Path to checkpoint .pt file to resume training")
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help="Directory to save model checkpoints")
    parser.add_argument('--num_frames', type=int, default=16, help="Number of frames per video clip")
    parser.add_argument('--k', type=int, default=5, help="Top k parameter for anomaly loss")
    parser.add_argument('--gamma', type=float, default=1.0, help="Temperature/scale parameter for cross entropy")
    parser.add_argument('--lambda_nce', type=float, default=1.0, help="Weight factor for classification loss")
    
    args = parser.parse_args()
    train_aadc_net(args)

if __name__ == "__main__":
    main()