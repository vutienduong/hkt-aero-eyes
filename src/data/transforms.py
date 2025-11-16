import cv2
import numpy as np
import torch
from torchvision import transforms as T
import random

class AeroEyesTransform:
    """
    Transform for Siamese tracking dataset.
    Applies augmentations to template and search images.
    """

    def __init__(
        self,
        template_size=(128, 128),
        search_size=(640, 360),
        mean=(0.485, 0.456, 0.406),  # ImageNet mean
        std=(0.229, 0.224, 0.225),   # ImageNet std
        color_jitter=True,
        horizontal_flip=True,
        blur=True,
    ):
        self.template_size = template_size
        self.search_size = search_size
        self.mean = mean
        self.std = std
        self.color_jitter = color_jitter
        self.horizontal_flip = horizontal_flip
        self.blur = blur

        # Color jitter augmentation
        if color_jitter:
            self.color_aug = T.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
            )

    def __call__(self, template_img, search_img, target_bbox):
        """
        Args:
            template_img: numpy array (H, W, 3) in RGB, uint8
            search_img: numpy array (H, W, 3) in RGB, uint8
            target_bbox: numpy array [x1, y1, x2, y2]

        Returns:
            template_tensor: (3, H, W) float32 tensor, normalized
            search_tensor: (3, H, W) float32 tensor, normalized
            target_bbox: (4,) float32 tensor [x1, y1, x2, y2], scaled to search_size
        """

        # Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            template_img = cv2.flip(template_img, 1)
            search_img = cv2.flip(search_img, 1)

            # Flip bbox coordinates
            H, W = search_img.shape[:2]
            x1, y1, x2, y2 = target_bbox
            target_bbox = np.array([W - x2, y1, W - x1, y2], dtype=np.float32)

        # Apply color jitter to search image only (simulate lighting changes)
        if self.color_jitter:
            search_img_pil = T.ToPILImage()(search_img)
            search_img_pil = self.color_aug(search_img_pil)
            search_img = np.array(search_img_pil)

        # Random blur (simulate motion blur)
        if self.blur and random.random() > 0.7:
            kernel_size = random.choice([3, 5])
            search_img = cv2.GaussianBlur(search_img, (kernel_size, kernel_size), 0)

        # Resize images
        template_resized = cv2.resize(template_img, self.template_size)
        search_resized = cv2.resize(search_img, self.search_size)

        # Scale bbox to new search size
        H_orig, W_orig = search_img.shape[:2]
        scale_x = self.search_size[0] / W_orig
        scale_y = self.search_size[1] / H_orig

        x1, y1, x2, y2 = target_bbox
        target_bbox_scaled = np.array([
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y,
        ], dtype=np.float32)

        # Convert to tensors and normalize
        template_tensor = self._to_tensor_normalize(template_resized)
        search_tensor = self._to_tensor_normalize(search_resized)

        return template_tensor, search_tensor, torch.from_numpy(target_bbox_scaled)

    def _to_tensor_normalize(self, img):
        """Convert numpy image to normalized tensor."""
        # img is (H, W, 3) uint8, RGB
        img = img.astype(np.float32) / 255.0  # [0, 1]

        # Normalize
        img = (img - np.array(self.mean)) / np.array(self.std)

        # HWC -> CHW
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        return img


class InferenceTransform:
    """
    Transform for inference (no augmentation).
    """

    def __init__(
        self,
        size=(640, 360),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.size = size
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            img: numpy array (H, W, 3) in RGB, uint8

        Returns:
            tensor: (3, H, W) normalized tensor
        """
        # Resize
        img_resized = cv2.resize(img, self.size)

        # To tensor and normalize
        img = img_resized.astype(np.float32) / 255.0
        img = (img - np.array(self.mean)) / np.array(self.std)
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        return img
