import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import Counter

class AADCDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_frames=16):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.video_paths = []
        self.labels = []
        self.normal_categories = []
        self.abnormal_categories = []
        
        normal_dir = os.path.join(root_dir, "normal")
        if not os.path.exists(normal_dir):
            normal_dir = os.path.join(root_dir, "Normal")
            
        if os.path.exists(normal_dir):
            for category in sorted(os.listdir(normal_dir)):
                category_path = os.path.join(normal_dir, category)
                if os.path.isdir(category_path):
                    if category not in self.normal_categories:
                        self.normal_categories.append(category)

        abnormal_dir = os.path.join(root_dir, "abnormal")
        if not os.path.exists(abnormal_dir):
            abnormal_dir = os.path.join(root_dir, "Abnormal")
            
        if os.path.exists(abnormal_dir):
            for category in sorted(os.listdir(abnormal_dir)):
                category_path = os.path.join(abnormal_dir, category)
                if os.path.isdir(category_path):
                    if category not in self.abnormal_categories:
                        self.abnormal_categories.append(category)

        self.categories = self.normal_categories + self.abnormal_categories

        if os.path.exists(normal_dir):
            for category in self.normal_categories:
                category_path = os.path.join(normal_dir, category)
                label_idx = self.categories.index(category)
                for root, _, files in os.walk(category_path):
                    for video_file in files:
                        if video_file.endswith(('.mp4', '.avi', '.mkv', '.webm')):
                            self.video_paths.append(os.path.join(root, video_file))
                            self.labels.append(label_idx)

        if os.path.exists(abnormal_dir):
            for category in self.abnormal_categories:
                category_path = os.path.join(abnormal_dir, category)
                label_idx = self.categories.index(category)
                for root, _, files in os.walk(category_path):
                    for video_file in files:
                        if video_file.endswith(('.mp4', '.avi', '.mkv', '.webm')):
                            self.video_paths.append(os.path.join(root, video_file))
                            self.labels.append(label_idx)

        # Check if we loaded any videos
        if len(self.video_paths) == 0:
            raise RuntimeError(
                f"No video files found in root directory '{root_dir}'. "
                f"Please verify that '{normal_dir}' and/or '{abnormal_dir}' directories exist and contain video files."
            )

    @property
    def num_normal_classes(self):
        return len(self.normal_categories)

    def get_text_prompts(self):
        prompts = []
        for cat in self.normal_categories:
            prompts.append(f"a video of normal activity showing {cat}")
        for cat in self.abnormal_categories:
            prompts.append(f"a video of abnormal activity showing {cat}")
        return prompts

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine exact frames to decode and count their repetitions
        target_indices = torch.linspace(0, max(total_frames - 1, 0), self.num_frames).long().tolist()
        index_counts = Counter(target_indices)
        
        frames = []
        i = 0
        
        while cap.isOpened() and len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if i in index_counts:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame)
                if self.transform:
                    pil_image = self.transform(pil_image)
                # Duplicate the frame directly if linspace requires it (e.g. for short videos)
                for _ in range(index_counts[i]):
                    frames.append(pil_image)
            i += 1
        cap.release()
        
        # Ensure we return exactly self.num_frames
        frames = frames[:self.num_frames]
        
        while len(frames) < self.num_frames:
            if len(frames) > 0:
                last_frame = frames[-1]
                if torch.is_tensor(last_frame):
                    frames.append(last_frame.clone())
                else:
                    frames.append(last_frame.copy())
            else:
                raise ValueError(f"Failed to read any frames from {video_path}")
                
        video_tensor = torch.stack(frames)
        return video_tensor, torch.tensor(label, dtype=torch.long)