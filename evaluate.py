"""
Evaluation script for AADC-Net: computes AUC (UCF-Crime style) and AP
(XD-Violence style), mirroring paper Tables I-II.

IMPORTANT - READ BEFORE TRUSTING THE NUMBERS THIS PRINTS:
-----------------------------------------------------------
The paper's AUC/AP scores are computed at the FRAME level, against frame-level
ground-truth anomaly annotations (i.e. "frames 450-600 of this video are
abnormal"). UCF-Crime and XD-Violence ship these as separate annotation
files/timestamps.

`AADCDataset` (dataset.py), as currently written, only stores ONE label per
ENTIRE video clip -- it never loads per-frame temporal annotations. That means:

  - If your data only has video-level labels (the common case for a
    weakly-supervised setup, which is what the paper itself assumes during
    TRAINING), you can only honestly evaluate at the VIDEO level: one
    predicted score (e.g. mean or max anomaly probability across sampled
    frames) vs. one ground-truth label per video. This script's
    `evaluate_video_level()` does exactly that, and is what you should use
    unless you add frame-level annotations.

  - If you DO have frame-level temporal ground truth available (e.g. as a
    .txt/.json file alongside each video, the format UCF-Crime/XD-Violence
    ship), you need to load it and pass it in as `frame_gt` per video to
    `evaluate_frame_level()` to reproduce literal Table I/II numbers. This
    script will NOT fabricate frame-level ground truth from a video-level
    label -- doing so would silently produce a metric that looks like the
    paper's but is not measuring the same thing.

Bottom line: run `evaluate_video_level()` honestly today. Only use
`evaluate_frame_level()` once you've wired in real per-frame annotations.
"""
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


@torch.no_grad()
def evaluate_video_level(model, dataloader, device, num_normal_classes):
    """
    Video-level AUC/AP: one anomaly score per video (mean anomaly probability
    across the sampled frames) vs. one ground-truth label per video
    (0 = normal, 1 = abnormal, derived the same way train.py does it).

    This does NOT require frame-level annotations and is honest about what
    it measures -- it is a coarser metric than the paper's frame-level
    Tables I-II, but it is the metric your current dataset format supports.
    """
    model.eval()
    all_scores, all_labels = [], []

    text_prompts = dataloader.dataset.dataset.get_text_prompts() \
        if hasattr(dataloader.dataset, "dataset") else dataloader.dataset.get_text_prompts()

    for videos, labels in dataloader:
        videos = videos.to(device)
        anomaly_logits, _ = model(videos, text_prompts)
        anomaly_probs = torch.sigmoid(anomaly_logits)         # [batch, frames]
        video_score = anomaly_probs.mean(dim=1)                # [batch]

        gt_anomaly = (labels >= num_normal_classes).float()

        all_scores.extend(video_score.cpu().tolist())
        all_labels.extend(gt_anomaly.cpu().tolist())

    if len(set(all_labels)) < 2:
        raise ValueError(
            "evaluate_video_level: only one class present in this split's labels -- "
            "AUC/AP are undefined. Check your split (e.g. test set with no abnormal videos)."
        )

    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    return {"AUC": auc, "AP": ap, "n_videos": len(all_labels)}


@torch.no_grad()
def evaluate_frame_level(model, dataloader, device, frame_gt_lookup, video_paths):
    """
    Frame-level AUC/AP, matching paper Tables I-II methodology.

    Args:
        frame_gt_lookup: dict mapping video_path -> list/array of 0/1 labels,
            one per frame index actually sampled by the dataset (i.e. length
            must equal dataset.num_frames for that video). YOU must supply
            this from real frame-level annotations; this function will not
            invent it.
        video_paths: parallel list of video paths for each item the
            dataloader will yield, in the same order (e.g.
            `dataset.video_paths` if iterating without shuffling).

    Raises:
        KeyError if a video in the dataloader has no entry in frame_gt_lookup,
        rather than silently skipping it or fabricating a label.
    """
    model.eval()
    all_scores, all_labels = [], []
    text_prompts = dataloader.dataset.dataset.get_text_prompts() \
        if hasattr(dataloader.dataset, "dataset") else dataloader.dataset.get_text_prompts()

    idx = 0
    for videos, labels in dataloader:
        videos = videos.to(device)
        batch_size = videos.shape[0]
        anomaly_logits, _ = model(videos, text_prompts)
        anomaly_probs = torch.sigmoid(anomaly_logits).cpu()    # [batch, frames]

        for b in range(batch_size):
            vp = video_paths[idx]
            idx += 1
            if vp not in frame_gt_lookup:
                raise KeyError(
                    f"No frame-level ground truth provided for '{vp}'. "
                    f"Refusing to fabricate labels -- add this video to frame_gt_lookup."
                )
            gt = frame_gt_lookup[vp]
            scores = anomaly_probs[b].tolist()
            if len(gt) != len(scores):
                raise ValueError(
                    f"Frame-count mismatch for '{vp}': got {len(gt)} GT labels "
                    f"but {len(scores)} predicted scores."
                )
            all_scores.extend(scores)
            all_labels.extend(gt)

    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    return {"AUC": auc, "AP": ap, "n_frames": len(all_labels)}


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from dataset import AADCDataset
    from model import AADCNet
    from splits import stratified_video_split, print_split_summary

    parser = argparse.ArgumentParser(description="AADC-Net evaluation (video-level AUC/AP)")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = AADCDataset(root_dir=args.data_dir, transform=transform, num_frames=args.num_frames)
    train_ds, val_ds, test_ds = stratified_video_split(dataset)
    print_split_summary(dataset, train_ds, val_ds, test_ds)

    split_map = {'train': train_ds, 'val': val_ds, 'test': test_ds}
    eval_ds = split_map[args.split]
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    model = AADCNet(num_classes=len(dataset.categories)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    metrics = evaluate_video_level(model, eval_loader, device, dataset.num_normal_classes)
    print(f"\n[{args.split} split] Video-level AUC: {metrics['AUC']:.4f}  "
          f"AP: {metrics['AP']:.4f}  (n={metrics['n_videos']} videos)")
    print("\nNOTE: this is VIDEO-level AUC/AP, not the paper's frame-level Table I/II metric "
          "-- see the module docstring in evaluate.py for why, and how to upgrade to frame-level "
          "once you have per-frame temporal annotations.")
