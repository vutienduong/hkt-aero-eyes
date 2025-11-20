import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from .data.aeroeyes_dataset import AeroEyesDataset
from .data.transforms import AeroEyesTransform
from .models.siam_tracker import SiamTracker
from .utils.logging_utils import get_logger


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
    # Clamp predicted boxes to prevent invalid coordinates
    pred_boxes = pred_boxes.clamp(min=0)

    # Ensure x2 > x1 and y2 > y1
    pred_boxes[:, 2] = torch.max(pred_boxes[:, 2], pred_boxes[:, 0] + eps)
    pred_boxes[:, 3] = torch.max(pred_boxes[:, 3], pred_boxes[:, 1] + eps)

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

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    # Learning rate scheduler
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

    # Gradient accumulation settings
    gradient_accumulation_steps = cfg["train"].get("gradient_accumulation_steps", 1)
    if gradient_accumulation_steps > 1:
        logger.info(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
        logger.info(f"Effective batch size: {cfg['data']['batch_size'] * gradient_accumulation_steps}")

    # Training loop
    for epoch in range(cfg["train"]["epochs"]):
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

                # Total loss (scale by accumulation steps for proper averaging)
                loss = (cls_loss + bbox_loss) / gradient_accumulation_steps

            # Backward with gradient scaling for mixed precision
            scaler.scale(loss).backward()

            # Only step optimizer after accumulating gradients
            if (step + 1) % gradient_accumulation_steps == 0:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Check for NaN/inf in gradients
                has_nan_inf = False
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_inf = True
                            break

                if has_nan_inf:
                    logger.warning(f"NaN/inf detected in gradients at step {step}, skipping optimizer step")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

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

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"epoch_{epoch+1}.ckpt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
        }, checkpoint_path)

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
                loss = cls_loss + bbox_loss

                val_losses.append(loss.item())

    model.train()
    return sum(val_losses) / len(val_losses)
