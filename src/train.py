import torch
from torch.utils.data import DataLoader
from .data.aeroeyes_dataset import AeroEyesDataset
from .models.siam_tracker import SiamTracker
from .utils.logging_utils import get_logger

def run_training(cfg):
    logger = get_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_ds = AeroEyesDataset(
        root=cfg["data"]["root"],
        annotations_file=cfg["data"]["annotations_file"],
        split_file=cfg["data"]["train_split"],
        transforms=None,  # TODO: add transforms
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model = SiamTracker(
        backbone_name=cfg["model"]["backbone"],
        feature_dim=cfg["model"]["feature_dim"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # TODO: define cls + bbox loss
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for step, batch in enumerate(train_loader):
            # template, search, target_bbox = batch
            # template, search, target_bbox = template.to(device), ...
            # logits, bbox_pred = model(search, template_img=template)
            # loss = ...
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            pass

        # TODO: validation + checkpoint saving
        logger.info(f"Finished epoch {epoch+1}/{cfg['train']['epochs']}")
