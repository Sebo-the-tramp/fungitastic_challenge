import os
import sys
import cv2
import torch
import random
import numpy as np

from pathlib import Path
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from utils import read_segments, polygon_to_mask

PROJECT_ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2")
DATASET_ROOT = Path("/data0/sebastian.cavada/datasets/FungiTastic")
OUTPUT_ROOT = PROJECT_ROOT / "data_processed"
MODEL_NAME = os.environ.get("MODEL_NAME", "facebook/dinov2-with-registers-small")
OUTPUT_NAME = os.environ.get("OUTPUT_NAME", "")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
SHARD_SIZE = int(os.environ.get("SHARD_SIZE", "512"))
SEED = 0
DTYPE = torch.bfloat16

sys.path.append(str(PROJECT_ROOT / "FungiTastic"))
from dataset.mask_fungi import MaskFungiTastic
from dataset.utils.mask_vis import get_image_shape, resize_mask_to_image

HUGGINGFACE_MODELS = [    
    "facebook/dinov2-with-registers-small",
    "facebook/dinov2-with-registers-base",
    "facebook/dinov2-with-registers-large",
    "facebook/dinov2-with-registers-giant",
]

SPLIT = os.environ.get("SPLIT", "train")
DEFAULT_BATCH_SIZE = 8
MODEL_LOAD_DTYPE = DTYPE
FEATURE_DTYPE = DTYPE
DATA_SUBSET = "all"
DATASET_SIZE = os.environ.get("DATASET_SIZE", "720")
IMAGE_SIZE = 224 if DATASET_SIZE == "300" else 448
TASK = "closed"
SEG_TASK = "binary"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = DEVICE == "cuda"
BACKGROUND_TYPES = [
    "normal",
]
BACKGROUND = "normal"
MIN_BOX_AREA = 2500

USE_SAM_MASKS = True
SAM_MASK_DIR = PROJECT_ROOT / "data_processed" / "sam3_yolo_generic_mushroom_200" / "all" / SPLIT / "720" / "FungiTastic" / SPLIT / "720p"

def load_sam_mask_batched(images_path:list[str], images:list[Image]) -> np.ndarray:

    sam_masks = []

    for image_path, image in zip(images_path, images):
        image_id = Path(image_path).stem
        img_width, img_height = image.size
        
        segment_path = SAM_MASK_DIR / f"{image_id}.txt"
        segments, _ = read_segments(segment_path)

        if segments is not None:
            bit_mask = polygon_to_mask(segments, img_width, img_height)
            sam_masks.append(torch.tensor(bit_mask.astype(bool)).unsqueeze(0))  # Add channel dimension for compatibility
        else:
            print(f"No SAM segments found for {image_id}, using empty mask.")
            raise NotImplementedError("SAM mask processing is not implemented for missing segments. Please ensure SAM masks are generated for all images or handle missing cases appropriately.")            

    return sam_masks

def collate_batch(batch):
    images = [item[0] for item in batch]
    # from /home/cavadalab/Documents/scsv/fungitastic2026/FungiTastic/dataset/mask_fungi.py
    masks = [torch.from_numpy(resize_mask_to_image(item[1], get_image_shape(item[0]))).unsqueeze(0) for item in batch]    
    labels = [item[2] for item in batch]
    file_paths = [item[3] for item in batch]
    return images, masks, labels, file_paths

def to_storage_feature(feature: torch.Tensor) -> torch.Tensor:
    return feature.to(device="cpu", dtype=FEATURE_DTYPE).contiguous()


def flush_shard(
    output_dir: Path,
    shard_index: int,
    file_paths: list[str],
    labels: list[int | None],
    cls_token: list[torch.Tensor],
    register_tokens: list[torch.Tensor] | None,
    mean_pooled_patch_tokens: list[torch.Tensor],
    mean_pooled_gt_masked_patch_tokens: list[torch.Tensor],
    mean_pooled_sam_masked_patch_tokens: list[torch.Tensor],
    patch_feature_batches: list[torch.Tensor], #optional, can be empty if not saving patch features    
) -> int:
    if not file_paths:
        return shard_index

    output_dir.mkdir(parents=True, exist_ok=True)
    shard = {
        "model_name": MODEL_NAME,
        "split": SPLIT,
        "feature_dtype": str(FEATURE_DTYPE),
        "file_paths": list(file_paths),
        "labels": [int(label) if label is not None else None for label in labels],
        "cls_token": torch.cat(cls_token, dim=0).contiguous(),
        "register_tokens": torch.cat(register_tokens, dim=0).contiguous() if register_tokens else None,
        "mean_pooled_patch_tokens": torch.cat(mean_pooled_patch_tokens, dim=0).contiguous(), 
        "mean_pooled_gt_masked_patch_tokens": torch.cat(mean_pooled_gt_masked_patch_tokens, dim=0).contiguous(),
        "mean_pooled_sam_masked_patch_tokens": torch.cat(mean_pooled_sam_masked_patch_tokens, dim=0).contiguous(),
        "patch_features": torch.cat(patch_feature_batches, dim=0).contiguous() if patch_feature_batches else None,
    }
    torch.save(shard, output_dir / f"shard_{shard_index:05d}.pt")
    file_paths.clear()
    labels.clear()
    cls_token.clear()
    mean_pooled_patch_tokens.clear()
    mean_pooled_gt_masked_patch_tokens.clear()
    mean_pooled_sam_masked_patch_tokens.clear()
    patch_feature_batches.clear()
    register_tokens.clear() if register_tokens else None

    return shard_index + 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_bounding_boxes(bool_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    uint8_mask = (bool_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(uint8_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [tuple(map(int, cv2.boundingRect(contour))) for contour in contours]
    assert bounding_boxes
    bounding_boxes = sorted(bounding_boxes, key=lambda box: (box[1], box[0], box[3], box[2]))
    filtered_boxes = [box for box in bounding_boxes if box[2] * box[3] >= MIN_BOX_AREA]
    return filtered_boxes if filtered_boxes else [max(bounding_boxes, key=lambda box: box[2] * box[3])]


def pre_process_images(images: list[Image.Image], masks: list[list[bool]], labels: list[int], background: str) -> list[Image.Image]:
    if background == "normal":
        return images, labels
    
    elif background == "masked_black":
        masked_images = []
        for img, msk in zip(images, masks):
            msk_uint8 = msk.to(torch.uint8) * 255
            mask_pil = TF.to_pil_image(msk_uint8).convert("L")
            black_bg = Image.new(img.mode, img.size, 0) 
            masked_img = Image.composite(img, black_bg, mask_pil)
            masked_images.append(masked_img)
        return masked_images, labels
    
    elif background == "masked_blurred":
        masked_images = []
        for img, msk in zip(images, masks):
            msk_uint8 = msk.to(torch.uint8) * 255

            # 2. Convert directly to a PIL Image 
            mask_pil = TF.to_pil_image(msk_uint8).convert("L")
            
            # 3. Create the blurred background
            blurred_bg = img.filter(ImageFilter.GaussianBlur(radius=15))
            
            # 4. Composite! 
            masked_img = Image.composite(img, blurred_bg, mask_pil)

            masked_images.append(masked_img)
        return masked_images, labels

    elif background == "crop":

        cropped_imges = []
        new_labels = []
        for img, msk, label in zip(images, masks, labels):
            bounding_boxes = get_bounding_boxes(np.array(msk.squeeze(0), dtype=bool))
            for box in bounding_boxes[:2]:  # Limit to the two largest boxes
                x, y, w, h = box
                cropped_img = img.crop((x, y, x + w, y + h))
                cropped_imges.append(cropped_img)
                new_labels.append(label)

        return cropped_imges, new_labels

    else:
        raise ValueError(f"Unknown background type: {background}")


def compute_mean_pooled_patch_tokens(patch_features: torch.Tensor) -> torch.Tensor:
    batch_size, num_patches_h, num_patches_w, feature_dim = patch_features.shape
    mean_pooled = patch_features.mean(dim=(1, 2))
    return mean_pooled

def compute_mean_pooled_masked_patch_tokens(
    patch_features: torch.Tensor,   # [B, H_p, W_p, D]
    masks: list[torch.Tensor],      # each [H, W]
    patch_size: int,
    image_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    B, H_p, W_p, D = patch_features.shape

    # print(f"Mask shape 0 {masks[0].shape}, expected ({image_size[1]}, {image_size[0]})")

    masks_resized = [torch.from_numpy(resize_mask_to_image(mask[0].numpy(), image_size)) for mask in masks]

    masks_tensor = torch.stack(masks_resized).float().unsqueeze(1)  # [B, 1, H, W]

    # reuse resize_mask_to_image
    #masks = [torch.from_numpy(resize_mask_to_image(item[1], get_image_shape(item[0]))).unsqueeze(0) for item in batch]    

    # Best if original mask resolution matches image resolution
    # and image size is divisible by patch_size
    patch_weights = F.avg_pool2d(
        masks_tensor,
        kernel_size=patch_size,
        stride=patch_size,
    )  # [B, 1, H_p, W_p]

    # Fallback safety if sizes are slightly off
    if patch_weights.shape[-2:] != (H_p, W_p):
        patch_weights = F.interpolate(
            patch_weights, size=(H_p, W_p), mode="area"
        )

    patch_weights = patch_weights.permute(0, 2, 3, 1).to('cuda')  # [B, H_p, W_p, 1]

    weighted_sum = (patch_features * patch_weights).sum(dim=(1, 2))  # [B, D]
    weight_sum = patch_weights.sum(dim=(1, 2)).clamp(min=1e-6)       # [B, 1]

    return weighted_sum / weight_sum


def main() -> None:
    seed_everything(SEED)
    # torch.set_grad_enabled(False)
    
    dataset = MaskFungiTastic(
        root=str(DATASET_ROOT),
        split=SPLIT,
        size=DATASET_SIZE,
        task=TASK,
        data_subset=DATA_SUBSET,
        transform=None,
        seg_task=SEG_TASK,
        workers=8,
    )
   
    assert MODEL_NAME in HUGGINGFACE_MODELS, MODEL_NAME

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, size={"height": IMAGE_SIZE, "width": IMAGE_SIZE})

    # Keep the pretrained config image_size. Dinov2 interpolates positional
    # embeddings at forward time for resized inputs like 448x448.
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=MODEL_LOAD_DTYPE,
    ).eval()
    dataloader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "collate_fn": collate_batch,
    }
    if NUM_WORKERS > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    save_patch_features = False
    patch_suffix = "_patches" if save_patch_features else ""
    output_model_root = OUTPUT_ROOT / MODEL_NAME / f"{DTYPE}_{BACKGROUND}_{IMAGE_SIZE}{patch_suffix}".replace("torch.", "") / SPLIT
    shard_index = 0
    shard_file_paths: list[str] = []
    shard_labels: list[int | None] = []
    shard_cls_tokens_batches: list[torch.Tensor] = []
    shard_register_tokens_batches: list[torch.Tensor] = []  # only for models with register tokens, otherwise will be empty and not saved
    shard_patch_feature_batches: list[torch.Tensor] = []
    shard_mean_pooled_patch_tokens: list[torch.Tensor] = []
    shard_mean_pooled_gt_masked_patch_tokens: list[torch.Tensor] = []
    shard_mean_pooled_sam_masked_patch_tokens: list[torch.Tensor] = []

    buffered_items = 0

    with torch.inference_mode():
        for images, gt_masks, labels, file_paths in tqdm(dataloader, desc="Processing batches", unit="batch"):

            images, labels = pre_process_images(images, gt_masks, labels, BACKGROUND)
            # print(images[0].size)
            itorchuts = processor(images=images, return_tensors="pt")
            # print(itorchuts["pixel_values"].shape)
            
            moved_itorchuts = {}
            for key, value in itorchuts.items():
                if PIN_MEMORY:
                    value = value.pin_memory()
                if value.is_floating_point():
                    moved_itorchuts[key] = value.to(
                        model.device,
                        dtype=MODEL_LOAD_DTYPE,
                        non_blocking=PIN_MEMORY,
                    )
                else:
                    moved_itorchuts[key] = value.to(
                        model.device,
                        non_blocking=PIN_MEMORY,
                    )
            itorchuts = moved_itorchuts
            outputs = model(**itorchuts)
            cls_tokens = to_storage_feature(outputs.pooler_output)
            last_hidden_states = outputs["last_hidden_state"]

            patch_size = model.config.patch_size
            _, _, img_height, img_width = itorchuts["pixel_values"].shape
            num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
            register_tokens = last_hidden_states[:, 1:1+model.config.num_register_tokens, :] if model.config.num_register_tokens > 0 else None
            patch_features_flat = last_hidden_states[:, 1 + model.config.num_register_tokens:, :]
            patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))   
            
            register_tokens = to_storage_feature(register_tokens) if register_tokens is not None else None
            shard_register_tokens_batches.append(register_tokens)

            # computing mean_pooled_patch_tokens
            mean_pooled_patch_tokens = compute_mean_pooled_patch_tokens(patch_features)

            # computing mean_pooled_gt_masked_patch_tokens
            # there seems to be something wrong with the img_width/img_heid
            mean_pooled_gt_masked_patch_tokens = compute_mean_pooled_masked_patch_tokens(patch_features, gt_masks, patch_size, image_size=(img_width, img_height))

            if USE_SAM_MASKS:
                sam_mask = load_sam_mask_batched(file_paths, images)  # Implement this function to get SAM masks for the images
                mean_pooled_sam_masked_patch_tokens = compute_mean_pooled_masked_patch_tokens(patch_features, sam_mask, patch_size, image_size=(img_width, img_height))
            else:
                raise NotImplementedError("SAM mask processing is not implemented in this version. Set USE_SAM_MASKS to False or implement the SAM mask loading and processing.")
                mean_pooled_sam_masked_patch_tokens = torch.zeros_like(mean_pooled_gt_masked_patch_tokens)  # Placeholder if not using SAM masks

            # saving or not full patch features (heavy on storage, but useful for future analysis)
            if save_patch_features:
                patches_features = to_storage_feature(patch_features)
            else:
                patches_features = None
        
            shard_file_paths.extend(file_paths)
            shard_labels.extend(int(label) if label is not None else None for label in labels)
            shard_cls_tokens_batches.append(cls_tokens)
            shard_mean_pooled_patch_tokens.append(mean_pooled_patch_tokens)
            shard_mean_pooled_gt_masked_patch_tokens.append(mean_pooled_gt_masked_patch_tokens)
            shard_mean_pooled_sam_masked_patch_tokens.append(mean_pooled_sam_masked_patch_tokens)  # placeholder for future SAM masked features, not computed in this version

            if patches_features is not None:
                shard_patch_feature_batches.append(patches_features)
            buffered_items += len(file_paths)

            if buffered_items >= SHARD_SIZE:
                shard_index = flush_shard(
                    output_dir=output_model_root,
                    shard_index=shard_index,
                    file_paths=shard_file_paths,
                    labels=shard_labels,
                    cls_token=shard_cls_tokens_batches,
                    register_tokens=shard_register_tokens_batches,
                    mean_pooled_patch_tokens=shard_mean_pooled_patch_tokens,
                    mean_pooled_gt_masked_patch_tokens=shard_mean_pooled_gt_masked_patch_tokens,
                    mean_pooled_sam_masked_patch_tokens=shard_mean_pooled_sam_masked_patch_tokens,
                    patch_feature_batches=shard_patch_feature_batches,
                )
                buffered_items = 0

    flush_shard(
        output_dir=output_model_root,
        shard_index=shard_index,
        file_paths=shard_file_paths,
        labels=shard_labels,
        cls_token=shard_cls_tokens_batches,
        register_tokens=shard_register_tokens_batches,
        mean_pooled_patch_tokens=shard_mean_pooled_patch_tokens,
        mean_pooled_gt_masked_patch_tokens=shard_mean_pooled_gt_masked_patch_tokens,
        mean_pooled_sam_masked_patch_tokens=shard_mean_pooled_sam_masked_patch_tokens,
        patch_feature_batches=shard_patch_feature_batches,
    )


if __name__ == "__main__":    
    main()
