from abc import ABC, abstractmethod
import torch
import numpy as np
import sys
import os
import sys

# -------------------------------------------------------
# Base class for all 3D models used in the unified pipeline.
# Every derived model must implement:
#   - load(): load model weights
#   - infer_chunk(): perform inference on a list of images
# -------------------------------------------------------

class Base3DModel(ABC):
    def __init__(self, config, device="cuda"):
        """
        Base class constructor.
        Args:
            config (dict): Configuration dictionary containing model paths/settings.
            device (str): Device to place the model on ("cuda" or "cpu").
        """
        self.config = config
        self.device = device
        # Automatically select bfloat16 for newer GPUs (SM >= 8), otherwise use float16.
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.model = None

    @abstractmethod
    def load(self):
        """Load model weights and initialize the model instance."""
        pass

    @abstractmethod
    def infer_chunk(self, image_paths: list) -> dict:
        """
        The unified inference interface used by all 3D models.
        Args:
            image_paths (list): List of image file paths.

        Returns:
            dict containing:
                - world_points: Predicted 3D points (B, N, 3)
                - world_points_conf: Confidence scores
                - extrinsic: Camera extrinsics (C2W)
                - intrinsic: Camera intrinsics
                - depth: Depth maps (optional)
                - depth_conf: Depth confidence (optional)
                - images: Preprocessed input images
                - mask: Optional mask
        """
        pass


# ===================== VGGT Adapter =====================
# Adapts the VGGT model to the unified Base3DModel interface.
# ========================================================

class VGGTAdapter(Base3DModel):
    def load(self):
        """Load VGGT model and its weights into memory."""
        print('Loading VGGT model...')
        from base_models.vggt.models.vggt import VGGT

        # Initialize model
        self.model = VGGT()

        # Load weights specified in config
        url = self.config['Weights']['VGGT']
        print(f"Loading weights from: {url}")
        state_dict = torch.load(url, map_location='cuda')

        # Strict=False allows missing/unmatched keys without crashing
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.model = self.model.to(self.device)

    def infer_chunk(self, image_paths: list) -> dict:
        """
        Run inference on a list of images using VGGT.
        Handles both normal mode and "middle reference frame" mode,
        which reorders and post-processes outputs for improved consistency.
        """
        from base_models.vggt.utils.load_fn import load_and_preprocess_images
        from base_models.vggt.utils.pose_enc import pose_encoding_to_extri_intri

        # Load images and preprocess them into a tensor: [B, 3, H, W]
        images = load_and_preprocess_images(image_paths).to(self.device)
        print(f"Loaded {len(images)} images")

        assert len(images.shape) == 4
        assert images.shape[1] == 3

        # Special mode: treat the middle frame as the reference image.
        if self.config['Model']['reference_frame_mid'] == True:
            torch.cuda.empty_cache()
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):

                    # Reorder so that the middle frame becomes the first
                    mid_idx = len(images) // 2
                    images = torch.cat(
                        [images[mid_idx:mid_idx + 1],  # middle frame
                         images[:mid_idx],             # frames before mid
                         images[mid_idx + 1:]          # frames after mid
                         ], dim=0)

                    # Run VGGT
                    predictions = self.model(images)

                    # Restore original ordering back to match input order.
                    # All predicted fields must be reordered consistently.
                    predictions["depth"] = torch.cat([
                        predictions["depth"][:, 1:mid_idx + 1],
                        predictions["depth"][:, :1],
                        predictions["depth"][:, mid_idx + 1:]
                    ], dim=1)

                    predictions["depth_conf"] = torch.cat([
                        predictions["depth_conf"][:, 1:mid_idx + 1],
                        predictions["depth_conf"][:, :1],
                        predictions["depth_conf"][:, mid_idx + 1:]
                    ], dim=1)

                    predictions["world_points"] = torch.cat([
                        predictions["world_points"][:, 1:mid_idx + 1],
                        predictions["world_points"][:, :1],
                        predictions["world_points"][:, mid_idx + 1:]
                    ], dim=1)

                    predictions["world_points_conf"] = torch.cat([
                        predictions["world_points_conf"][:, 1:mid_idx + 1],
                        predictions["world_points_conf"][:, :1],
                        predictions["world_points_conf"][:, mid_idx + 1:]
                    ], dim=1)

                    predictions["pose_enc"] = torch.cat([
                        predictions["pose_enc"][:, 1:mid_idx + 1],
                        predictions["pose_enc"][:, :1],
                        predictions["pose_enc"][:, mid_idx + 1:]
                    ], dim=1)

                    predictions["images"] = torch.cat([
                        predictions["images"][:, 1:mid_idx + 1],
                        predictions["images"][:, :1],
                        predictions["images"][:, mid_idx + 1:]
                    ], dim=1)

            torch.cuda.empty_cache()

        else:
            # Standard inference path
            torch.cuda.empty_cache()
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    predictions = self.model(images)
            torch.cuda.empty_cache()

        # Convert pose encoding into extrinsic (C2W) and intrinsic matrices
        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

        # The model outputs W2C; take inverse to get C2W
        # Convert 3x4 matrix [R|t] into 4x4 homogeneous matrix
        ones = torch.tensor([0, 0, 0, 1], dtype=extrinsic.dtype, device=extrinsic.device)
        ones = ones.view(1, 1, 1, 4).repeat(extrinsic.shape[0], extrinsic.shape[1], 1, 1)

        # Concatenate to make (B, N, 4, 4)
        extrinsic_homo = torch.cat([extrinsic, ones], dim=2)

        # Inverse to get C2W
        predictions["extrinsic"] = torch.inverse(extrinsic_homo)
        predictions["intrinsic"] = intrinsic
        ##point : torch.Size([1, 60, 154, 518, 3])
        ##conf : torch.Size([1, 60, 154, 518])
        ##depth : torch.Size([1, 60, 154, 518, 1])

        return {
            'world_points': predictions["world_points"],
            'world_points_conf': predictions["world_points_conf"],
            'extrinsic': predictions["extrinsic"],
            'intrinsic': predictions["intrinsic"],
            'depth': predictions["depth"],
            'depth_conf': predictions["depth_conf"],
            'images': predictions["images"],
            'mask': None
        }


# ===================== Pi3 Adapter =====================
# Adapts Pi3 to the unified 3D inference interface.
# =======================================================

class Pi3Adapter(Base3DModel):
    def load(self):
        """Load Pi3 model and its safetensors weights."""
        print('Loading Pi3 model...')
        from base_models.pi3.models.pi3 import Pi3
        from safetensors.torch import load_file

        # Initialize model
        self.model = Pi3().to(self.device).eval()

        # Load weights from safetensors
        url = self.config['Weights']['Pi3']
        print(f"Loading weights from: {url}")
        weight = load_file(url)
        self.model.load_state_dict(weight, strict=False)

    def infer_chunk(self, image_paths: list) -> dict:
        """
        Run inference with Pi3 on a list of images.
        Pi3 expects batched input of shape [1, B, 3, H, W].
        """
        from base_models.pi3.utils.basic import load_images_as_tensor_pi_long

        # Load images as Tensor: [B, 3, H, W]
        images = load_images_as_tensor_pi_long(image_paths).to(self.device)
        print(f"Loaded {len(images)} images")

        assert len(images.shape) == 4
        assert images.shape[1] == 3

        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                # Pi3 expects input as [1, B, ...]
                predictions = self.model(images[None])

        # Attach original images into predictions
        predictions['images'] = images[None]

        # Fix confidence: apply sigmoid (as per official Pi3 issue #55)
        conf = predictions['conf']
        conf = torch.sigmoid(conf)
        predictions['conf'] = conf.squeeze(-1)

        torch.cuda.empty_cache()

        return {
            'world_points': predictions['points'],
            'world_points_conf': predictions['conf'],
            'extrinsic': predictions['camera_poses'],  # already C2W
            'intrinsic': None,
            'depth': None,
            'depth_conf': None,
            'images': predictions['images'],
            'mask': None
        }
