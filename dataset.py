import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

class AADCDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_sec=30, num_frames=16):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_sec = frames_per_sec
        self.num_frames = num_frames
        self.video_paths = []
        self.labels = []
        self.categories = ["normal"]
        
        # Load normal videos
        normal_dir = os.path.join(root_dir, "normal")
        if not os.path.exists(normal_dir):
            normal_dir = os.path.join(root_dir, "Normal")
            
        if os.path.exists(normal_dir):
            for root, _, files in os.walk(normal_dir):
                for video_file in files:
                    if video_file.endswith(('.mp4', '.avi', '.mkv', '.webm')):
                        self.video_paths.append(os.path.join(root, video_file))
                        self.labels.append(0)

        # Load abnormal videos from subdirectories
        abnormal_dir = os.path.join(root_dir, "abnormal")
        if not os.path.exists(abnormal_dir):
            abnormal_dir = os.path.join(root_dir, "Abnormal")
            
        if os.path.exists(abnormal_dir):
            for category in os.listdir(abnormal_dir):
                category_path = os.path.join(abnormal_dir, category)
                if os.path.isdir(category_path):
                    if category not in self.categories:
                        self.categories.append(category)
                    label_idx = self.categories.index(category)
                    for root, _, files in os.walk(category_path):
                        for video_file in files:
                            if video_file.endswith(('.mp4', '.avi', '.mkv', '.webm')):
                                self.video_paths.append(os.path.join(root, video_file))
                                self.labels.append(label_idx)

    def get_text_prompts(self):
        prompts = ["a video of normal activities"]
        for cat in self.categories[1:]:
            prompts.append(f"a video of abnormal activity showing {cat}")
        return prompts

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        if total_frames > 0:
            indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx.item())
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame)
                    if self.transform:
                        pil_image = self.transform(pil_image)
                    frames.append(pil_image)
        cap.release()
        
        # Pad if necessary
        while len(frames) < self.num_frames:
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                frames.append(torch.zeros((3, 224, 224)))
                
        video_tensor = torch.stack(frames) if isinstance(frames[0], torch.Tensor) else torch.zeros((self.num_frames, 3, 224, 224))
        # Ensure we return a single tensor (e.g. [T, C, H, W])
        return video_tensor, torch.tensor(label, dtype=torch.long)