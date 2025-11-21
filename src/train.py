import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import re
import glob
from .data.aeroeyes_dataset import AeroEyesDataset
from .data.transforms import AeroEyesTransform
from .models.siam_tracker import SiamTracker
from .utils.logging_utils import get_logger


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    # Find all checkpoint files matching pattern: checkpoint_epoch{e}_step{s}.ckpt
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch*_step*.ckpt"))
    if not checkpoints:
        return None

    # Parse epoch and step from filenames and find the latest
    latest_ckpt = None
    latest_epoch = -1
    latest_step = -1

    for ckpt_path in checkpoints:
        match = re.search(r'checkpoint_epoch(\d+)_step(\d+)\.ckpt', ckpt_path.name)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))
            if epoch > latest_epoch or (epoch == latest_epoch and step > latest_step):
                latest_epoch = epoch
                latest_step = step
                latest_ckpt = ckpt_path

    return latest_ckpt


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device, logger):
    """Load checkpoint and return start epoch and step."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch']
    start_step = checkpoint.get('global_step', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    logger.info(f"Resumed from Epoch {start_epoch}, Global Step {start_step}")
    logger.info(f"Best validation loss so far: {best_val_loss:.4f}")

    return start_epoch, start_step, best_val_loss


def save_checkpoint(checkpoint_dir, epoch, global_step, model, optimizer, scheduler,
                   train_loss, best_val_loss, logger, keep_last_n=3):
    """Save checkpoint and cleanup old ones."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}_step{global_step}.ckpt"

    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'best_val_loss': best_val_loss,
    }, checkpoint_path)

    logger.info(f"Saved checkpoint: {checkpoint_path.name}")

    # Cleanup old checkpoints, keep only last N
    if keep_last_n > 0:
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_epoch*_step*.ckpt"),
            key=lambda p: (
                int(re.search(r'epoch(\d+)', p.name).group(1)),
                int(re.search(r'step(\d+)', p.name).group(1))
            )
        )

        # Remove older checkpoints
        for old_ckpt in checkpoints[:-keep_last_n]:
            old_ckpt.unlink()
            logger.info(f"Removed old checkpoint: {old_ckpt.name}")


def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    Args:
        boxes1: (N, 4) tensor [x1, y1, x2, y2]
        boxes2: (N, 4) tensor [x1, y1, x2, y2]
    Returns:
        iou: (N,) tensor
    """
    x1_max = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1_max = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2_min = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2_min = torch.min(boxes1[:, 3], boxes2[:, 3])

    intersection = torch.clamp(x2_min - x1_max, min=0) * torch.clamp(y2_min - y1_max, min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1 + area2 - intersection

    iou = intersection / (union + 1e-6)
    return iou


def giou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Generalized IoU loss with improved numerical stability.
    Args:
        pred_boxes: (N, 4) [x1, y1, x2, y2]
        target_boxes: (N, 4) [x1, y1, x2, y2]
    Returns:
        loss: scalar
    """
    # Clone to avoid in-place operations on tensors that need gradients
    pred_boxes = pred_boxes.clone()

    # Clamp predicted boxes to prevent invalid coordinates
    pred_boxes = pred_boxes.clamp(min=0)

    # Ensure x2 > x1 and y2 > y1 (avoiding in-place ops)
    pred_boxes = torch.stack([
        pred_boxes[:, 0],
        pred_boxes[:, 1],
        torch.max(pred_boxes[:, 2], pred_boxes[:, 0] + eps),
        torch.max(pred_boxes[:, 3], pred_boxes[:, 1] + eps)
    ], dim=1)

    # IoU
    iou = compute_iou(pred_boxes, target_boxes)

    # Smallest enclosing box
    x1_min = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1_min = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2_max = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2_max = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

    enclosing_area = torch.clamp((x2_max - x1_min) * (y2_max - y1_min), min=eps)

    # Union area (corrected calculation)
    area1 = torch.clamp((pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1]), min=eps)
    area2 = torch.clamp((target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1]), min=eps)

    # Compute intersection for union
    x1_max = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1_max = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2_min = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2_min = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    intersection = torch.clamp(x2_min - x1_max, min=0) * torch.clamp(y2_min - y1_max, min=0)

    union = area1 + area2 - intersection

    # GIoU with numerical stability
    giou = iou - torch.clamp((enclosing_area - union) / enclosing_area, min=-1.0, max=1.0)

    # Loss is 1 - GIoU, clamped to [0, 2]
    loss = torch.clamp(1 - giou, min=0.0, max=2.0)
    return loss.mean()


def run_training(cfg):
    logger = get_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Enable cuDNN benchmarking for faster training on A100
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmarking enabled for GPU optimization")

    # Setup Automatic Mixed Precision (AMP) for faster training
    from torch.amp import autocast, GradScaler
    scaler = GradScaler('cuda')

    # Create transforms
    train_transform = AeroEyesTransform(
        template_size=tuple(cfg["model"]["template_size"]),
        search_size=tuple(cfg["data"]["frame_size"]),
        color_jitter=True,
        horizontal_flip=True,
        blur=True,
    )

    # Training dataset
    train_ds = AeroEyesDataset(
        root=cfg["data"]["root"],
        annotations_file=cfg["data"]["annotations_file"],
        split_file=cfg["data"]["train_split"],
        transforms=train_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=True if cfg["data"]["num_workers"] > 0 else False,
        prefetch_factor=4 if cfg["data"]["num_workers"] > 0 else None,
    )

    # Validation dataset
    val_ds = AeroEyesDataset(
        root=cfg["data"]["root"],
        annotations_file=cfg["data"]["annotations_file"],
        split_file=cfg["data"]["val_split"],
        transforms=train_transform,  # Use same transform for now
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=True if cfg["data"]["num_workers"] > 0 else False,
        prefetch_factor=4 if cfg["data"]["num_workers"] > 0 else None,
    )

    logger.info(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    # Model
    model = SiamTracker(
        backbone_name=cfg["model"]["backbone"],
        feature_dim=cfg["model"]["feature_dim"],
    ).to(device)

    # Optimizer (start with low LR for warmup)
    warmup_steps = cfg["train"].get("lr_warmup_steps", 0)
    target_lr = float(cfg["train"]["lr"])
    initial_lr = target_lr * 0.1 if warmup_steps > 0 else target_lr

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    # Learning rate scheduler with warmup
    if warmup_steps > 0:
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup from 0.1x to 1.0x
                return 0.1 + (0.9 * current_step / warmup_steps)
            else:
                # Constant LR after warmup
                return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        logger.info(f"Using warmup scheduler: {warmup_steps} steps from {initial_lr:.2e} to {target_lr:.2e}")
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["train"].get("lr_step", 10),
            gamma=cfg["train"].get("lr_gamma", 0.1),
        )

    # Loss functions
    cls_criterion = nn.BCEWithLogitsLoss()

    # Checkpoint directory
    checkpoint_dir = Path(cfg["train"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    start_epoch = 0
    global_step = 0

    # Check for existing checkpoints and resume if requested
    resume_mode = cfg["train"].get("resume_from_checkpoint", None)
    if resume_mode:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)

        if latest_ckpt:
            if resume_mode == "auto":
                # Ask user if they want to resume
                try:
                    match = re.search(r'checkpoint_epoch(\d+)_step(\d+)\.ckpt', latest_ckpt.name)
                    if match:
                        ckpt_epoch = int(match.group(1))
                        ckpt_step = int(match.group(2))
                        logger.info(f"Found checkpoint: {latest_ckpt.name}")
                        logger.info(f"Resume from Epoch {ckpt_epoch}, Step {ckpt_step}?")
                        response = input("Enter 'y' to resume, 'n' to start fresh: ").strip().lower()

                        if response == 'y':
                            start_epoch, global_step, best_val_loss = load_checkpoint(
                                latest_ckpt, model, optimizer, scheduler, device, logger
                            )
                        else:
                            logger.info("Starting fresh training...")
                except Exception as e:
                    logger.warning(f"Error prompting for resume: {e}. Starting fresh...")

            elif resume_mode == "latest":
                # Auto-resume from latest
                start_epoch, global_step, best_val_loss = load_checkpoint(
                    latest_ckpt, model, optimizer, scheduler, device, logger
                )
        else:
            logger.info("No checkpoint found. Starting training from scratch...")

    # Gradient accumulation settings
    gradient_accumulation_steps = cfg["train"].get("gradient_accumulation_steps", 1)
    if gradient_accumulation_steps > 1:
        logger.info(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
        logger.info(f"Effective batch size: {cfg['data']['batch_size'] * gradient_accumulation_steps}")

    # Checkpoint save settings
    save_checkpoint_steps = cfg["train"].get("save_checkpoint_steps", 100)
    keep_last_n = cfg["train"].get("keep_last_n_checkpoints", 3)

    # Training loop
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        train_losses = []
        train_cls_losses = []
        train_bbox_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")

        for step, (template, search, target_bbox) in enumerate(pbar):
            # Use non_blocking for async GPU transfers
            template = template.to(device, non_blocking=True)
            search = search.to(device, non_blocking=True)
            target_bbox = target_bbox.to(device, non_blocking=True)  # (B, 4)

            # Wrap forward pass in autocast for mixed precision
            with autocast('cuda'):
                # Forward pass
                cls_logits, bbox_pred = model(search, template_img=template)

                # Classification loss
                # For simplicity, we'll use a heatmap approach
                # Create a target heatmap based on bbox location
                B, _, H, W = cls_logits.shape

                # Precompute meshgrid once per batch (not per sample)
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing='ij'
                )

                # Simple approach: center of bbox is positive, rest is negative
                target_cls = torch.zeros_like(cls_logits)
                sigma = 2.0
                for i in range(B):
                    x1, y1, x2, y2 = target_bbox[i]
                    cx = ((x1 + x2) / 2 / cfg["data"]["frame_size"][0] * W).long().clamp(0, W-1)
                    cy = ((y1 + y2) / 2 / cfg["data"]["frame_size"][1] * H).long().clamp(0, H-1)

                    # Create Gaussian around center (reusing precomputed grid)
                    gaussian = torch.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2))
                    target_cls[i, 0] = gaussian

                cls_loss = cls_criterion(cls_logits, target_cls)

                # Bbox regression loss
                # Get bbox prediction at center location
                bbox_preds_at_center = []
                for i in range(B):
                    x1, y1, x2, y2 = target_bbox[i]
                    cx = ((x1 + x2) / 2 / cfg["data"]["frame_size"][0] * W).long().clamp(0, W-1)
                    cy = ((y1 + y2) / 2 / cfg["data"]["frame_size"][1] * H).long().clamp(0, H-1)
                    bbox_preds_at_center.append(bbox_pred[i, :, cy, cx])

                bbox_preds_at_center = torch.stack(bbox_preds_at_center)  # (B, 4)

                # Scale predictions to image size
                bbox_preds_scaled = bbox_preds_at_center * torch.tensor(
                    [cfg["data"]["frame_size"][0], cfg["data"]["frame_size"][1],
                     cfg["data"]["frame_size"][0], cfg["data"]["frame_size"][1]],
                    device=device
                )

                # Compute GIoU loss
                bbox_loss = giou_loss(bbox_preds_scaled, target_bbox)

                # Check for NaN/inf in losses
                if torch.isnan(cls_loss) or torch.isinf(cls_loss):
                    logger.warning(f"NaN/inf in cls_loss at step {step}")
                    cls_loss = torch.tensor(0.0, device=device, requires_grad=True)

                if torch.isnan(bbox_loss) or torch.isinf(bbox_loss):
                    logger.warning(f"NaN/inf in bbox_loss at step {step}")
                    logger.warning(f"  pred_boxes range: [{bbox_preds_scaled.min():.2f}, {bbox_preds_scaled.max():.2f}]")
                    logger.warning(f"  target_boxes range: [{target_bbox.min():.2f}, {target_bbox.max():.2f}]")
                    bbox_loss = torch.tensor(0.0, device=device, requires_grad=True)

                # Total loss with bbox loss weighting (scale by accumulation steps for proper averaging)
                bbox_loss_weight = cfg["train"].get("bbox_loss_weight", 1.0)
                loss = (cls_loss + bbox_loss_weight * bbox_loss) / gradient_accumulation_steps

            # Backward with gradient scaling for mixed precision
            scaler.scale(loss).backward()

            # Only step optimizer after accumulating gradients
            if (step + 1) % gradient_accumulation_steps == 0:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)

                # Check gradient norms before clipping
                total_norm = 0
                max_grad = 0
                has_nan_inf = False
                for param in model.parameters():
                    if param.grad is not None:
                        # Check for NaN/inf
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_inf = True
                            break
                        # Track gradient statistics
                        param_norm = param.grad.data.norm(2).item()
                        total_norm += param_norm ** 2
                        max_grad = max(max_grad, param.grad.abs().max().item())

                total_norm = total_norm ** 0.5

                if has_nan_inf:
                    logger.warning(f"NaN/inf detected in gradients at step {step}")
                    logger.warning(f"  Loss values: cls={cls_loss.item():.4f}, bbox={bbox_loss.item():.4f}")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                else:
                    # Log gradient stats periodically
                    if step % 10 == 0:
                        logger.info(f"Gradient norm: {total_norm:.4f}, max grad: {max_grad:.4f}")

                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    # Step scheduler after optimizer update (for warmup)
                    if warmup_steps > 0:
                        scheduler.step()

                    # Increment global step and save checkpoint if needed
                    global_step += 1

                    if save_checkpoint_steps > 0 and global_step % save_checkpoint_steps == 0:
                        save_checkpoint(
                            checkpoint_dir, epoch + 1, global_step, model, optimizer, scheduler,
                            actual_loss, best_val_loss, logger, keep_last_n
                        )

            # Logging (multiply back to get actual loss for display)
            actual_loss = loss.item() * gradient_accumulation_steps
            train_losses.append(actual_loss)
            train_cls_losses.append(cls_loss.item())
            train_bbox_losses.append(bbox_loss.item())

            pbar.set_postfix({
                'loss': f'{actual_loss:.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'bbox': f'{bbox_loss.item():.4f}'
            })

            # Periodic logging
            if (step + 1) % cfg["train"]["log_interval"] == 0:
                logger.info(
                    f"Epoch {epoch+1}, Step {step+1}: "
                    f"loss={actual_loss:.4f}, cls={cls_loss.item():.4f}, bbox={bbox_loss.item():.4f}"
                )

        avg_train_loss = sum(train_losses) / len(train_losses)
        logger.info(
            f"Epoch {epoch+1} Training: "
            f"avg_loss={avg_train_loss:.4f}, "
            f"avg_cls={sum(train_cls_losses)/len(train_cls_losses):.4f}, "
            f"avg_bbox={sum(train_bbox_losses)/len(train_bbox_losses):.4f}"
        )

        # Validation
        if (epoch + 1) % cfg["train"]["val_interval"] == 0:
            val_loss = validate(model, val_loader, device, cls_criterion, cfg)
            logger.info(f"Epoch {epoch+1} Validation: avg_loss={val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = checkpoint_dir / "best_model.ckpt"
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")

        # Save epoch checkpoint
        save_checkpoint(
            checkpoint_dir, epoch + 1, global_step, model, optimizer, scheduler,
            avg_train_loss, best_val_loss, logger, keep_last_n
        )

        # Step scheduler only if not using warmup (warmup steps per batch)
        if warmup_steps == 0:
            scheduler.step()

    logger.info("Training complete!")


def validate(model, val_loader, device, cls_criterion, cfg):
    """Run validation."""
    from torch.amp import autocast

    model.eval()
    val_losses = []

    with torch.no_grad():
        for template, search, target_bbox in tqdm(val_loader, desc="Validation"):
            # Use non_blocking for async GPU transfers
            template = template.to(device, non_blocking=True)
            search = search.to(device, non_blocking=True)
            target_bbox = target_bbox.to(device, non_blocking=True)

            # Use autocast for mixed precision inference
            with autocast('cuda'):
                cls_logits, bbox_pred = model(search, template_img=template)

                # Same loss computation as training
                B, _, H, W = cls_logits.shape

                # Precompute meshgrid once per batch (not per sample)
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing='ij'
                )

                target_cls = torch.zeros_like(cls_logits)
                sigma = 2.0
                for i in range(B):
                    x1, y1, x2, y2 = target_bbox[i]
                    cx = ((x1 + x2) / 2 / cfg["data"]["frame_size"][0] * W).long().clamp(0, W-1)
                    cy = ((y1 + y2) / 2 / cfg["data"]["frame_size"][1] * H).long().clamp(0, H-1)

                    # Create Gaussian around center (reusing precomputed grid)
                    gaussian = torch.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2))
                    target_cls[i, 0] = gaussian

                cls_loss = cls_criterion(cls_logits, target_cls)

                bbox_preds_at_center = []
                for i in range(B):
                    x1, y1, x2, y2 = target_bbox[i]
                    cx = ((x1 + x2) / 2 / cfg["data"]["frame_size"][0] * W).long().clamp(0, W-1)
                    cy = ((y1 + y2) / 2 / cfg["data"]["frame_size"][1] * H).long().clamp(0, H-1)
                    bbox_preds_at_center.append(bbox_pred[i, :, cy, cx])

                bbox_preds_at_center = torch.stack(bbox_preds_at_center)
                bbox_preds_scaled = bbox_preds_at_center * torch.tensor(
                    [cfg["data"]["frame_size"][0], cfg["data"]["frame_size"][1],
                     cfg["data"]["frame_size"][0], cfg["data"]["frame_size"][1]],
                    device=device
                )

                bbox_loss = giou_loss(bbox_preds_scaled, target_bbox)

                # Apply same bbox loss weighting as training
                bbox_loss_weight = cfg["train"].get("bbox_loss_weight", 1.0)
                loss = cls_loss + bbox_loss_weight * bbox_loss

                val_losses.append(loss.item())

    model.train()
    return sum(val_losses) / len(val_losses)
