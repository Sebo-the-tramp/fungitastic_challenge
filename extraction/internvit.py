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
from transformers import AutoModel, CLIPImageProcessor
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2")
DATASET_ROOT = Path("/data0/sebastian.cavada/datasets/FungiTastic")
OUTPUT_ROOT = PROJECT_ROOT / "data_processed"
MODEL_NAME = os.environ.get("MODEL_NAME", "OpenGVLab/InternViT-6B-448px-V2_5")
OUTPUT_NAME = os.environ.get("OUTPUT_NAME", "")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
SHARD_SIZE = int(os.environ.get("SHARD_SIZE", "512"))
SEED = 0
DTYPE = torch.bfloat16

sys.path.append(str(PROJECT_ROOT / "FungiTastic"))
from dataset.mask_fungi import MaskFungiTastic

HUGGINGFACE_MODELS = ["OpenGVLab/InternViT-300M-448px-V2_5", "OpenGVLab/InternViT-6B-448px-V2_5"]

SPLIT = os.environ.get("SPLIT", "train")
MODEL_LOAD_DTYPE = DTYPE
FEATURE_DTYPE = DTYPE
DATA_SUBSET = "all"
DATASET_SIZE = os.environ.get("DATASET_SIZE", "720")
IMAGE_SIZE = 224 if DATASET_SIZE == "300" else 448
TASK = "closed"
SEG_TASK = "binary"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = DEVICE == "cuda"
BACKGROUND_TYPES = ["normal", "masked_black", "masked_blurred", "crop"]
BACKGROUND = os.environ.get("BACKGROUND", "normal")
MIN_BOX_AREA = 2500
SAVE_PATCH_TOKENS = os.environ.get("SAVE_PATCH_TOKENS", "0") == "1"
SAVE_HIDDEN_STATES = os.environ.get("SAVE_HIDDEN_STATES", "0") == "1"
ALLOW_OVERWRITE = os.environ.get("ALLOW_OVERWRITE", "0") == "1"


def collate_batch(batch: list[tuple[Image.Image, np.ndarray, int, str]]) -> tuple[list[Image.Image], list[torch.Tensor], list[int], list[str]]:
    images = [item[0] for item in batch]
    masks = [torch.from_numpy(item[1]).unsqueeze(0) for item in batch]
    labels = [item[2] for item in batch]
    file_paths = [item[3] for item in batch]
    return images, masks, labels, file_paths


def to_storage_feature(feature: torch.Tensor) -> torch.Tensor:
    return feature.to(device="cpu", dtype=FEATURE_DTYPE).contiguous()


def to_storage_hidden_states(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.stack([to_storage_feature(hidden_state) for hidden_state in hidden_states], dim=1)


def flush_shard(
    output_dir: Path,
    shard_index: int,
    file_paths: list[str],
    labels: list[int | None],
    cls_token: list[torch.Tensor],
    patch_tokens: list[torch.Tensor],
    hidden_states: list[torch.Tensor],
    patch_size: int,
    image_size: int,
) -> int:
    if not file_paths:
        return shard_index

    shard = {
        "model_name": MODEL_NAME,
        "split": SPLIT,
        "feature_dtype": str(FEATURE_DTYPE),
        "image_size": image_size,
        "patch_size": patch_size,
        "file_paths": list(file_paths),
        "labels": [int(label) if label is not None else None for label in labels],
        "cls_token": torch.cat(cls_token, dim=0).contiguous(),
        "patch_tokens": torch.cat(patch_tokens, dim=0).contiguous() if patch_tokens else None,
        "hidden_states": torch.cat(hidden_states, dim=0).contiguous() if hidden_states else None,
    }
    torch.save(shard, output_dir / f"shard_{shard_index:05d}.pt")
    file_paths.clear()
    labels.clear()
    cls_token.clear()
    patch_tokens.clear()
    hidden_states.clear()
    return shard_index + 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_bounding_boxes(bool_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    uint8_mask = (bool_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(uint8_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [tuple(map(int, cv2.boundingRect(contour))) for contour in contours]
    assert bounding_boxes
    bounding_boxes = sorted(bounding_boxes, key=lambda box: (box[1], box[0], box[3], box[2]))
    filtered_boxes = [box for box in bounding_boxes if box[2] * box[3] >= MIN_BOX_AREA]
    return filtered_boxes if filtered_boxes else [max(bounding_boxes, key=lambda box: box[2] * box[3])]


def pre_process_images(
    images: list[Image.Image],
    masks: list[torch.Tensor],
    labels: list[int | None],
    file_paths: list[str],
    background: str,
) -> tuple[list[Image.Image], list[int | None], list[str]]:
    assert background in BACKGROUND_TYPES, background

    if background == "normal":
        return images, labels, file_paths

    if background == "masked_black":
        masked_images = []
        for image, mask in zip(images, masks):
            mask_image = TF.to_pil_image(mask.to(torch.uint8) * 255).convert("L")
            masked_images.append(Image.composite(image, Image.new(image.mode, image.size, 0), mask_image))
        return masked_images, labels, file_paths

    if background == "masked_blurred":
        masked_images = []
        for image, mask in zip(images, masks):
            mask_image = TF.to_pil_image(mask.to(torch.uint8) * 255).convert("L")
            masked_images.append(Image.composite(image, image.filter(ImageFilter.GaussianBlur(radius=15)), mask_image))
        return masked_images, labels, file_paths

    cropped_images = []
    cropped_labels = []
    cropped_file_paths = []
    for image, mask, label, file_path in zip(images, masks, labels, file_paths):
        for x, y, w, h in get_bounding_boxes(np.array(mask.squeeze(0), dtype=bool))[:2]:
            cropped_images.append(image.crop((x, y, x + w, y + h)))
            cropped_labels.append(label)
            cropped_file_paths.append(file_path)
    return cropped_images, cropped_labels, cropped_file_paths


def prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = sorted(output_dir.glob("shard_*.pt"))
    if not shard_paths:
        return
    if not ALLOW_OVERWRITE:
        answer = input(f"{output_dir} already contains {len(shard_paths)} shard files. Overwrite? [y/N]: ").strip().lower()
        assert answer == "y", "Aborted to avoid overwriting existing shard files."
    for shard_path in shard_paths:
        shard_path.unlink()


def build_output_dir() -> Path:
    suffix = ""
    if SAVE_PATCH_TOKENS:
        suffix += "_patch_tokens"
    if SAVE_HIDDEN_STATES:
        suffix += "_hidden_states"
    run_name = OUTPUT_NAME or f"{DTYPE}_{BACKGROUND}_{IMAGE_SIZE}{suffix}".replace("torch.", "")
    return OUTPUT_ROOT / MODEL_NAME / run_name / SPLIT


def main() -> None:
    seed_everything(SEED)
    torch.set_grad_enabled(False)

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
    processor = CLIPImageProcessor.from_pretrained(
        MODEL_NAME,
        size={"shortest_edge": IMAGE_SIZE},
        crop_size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=MODEL_LOAD_DTYPE,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    if IMAGE_SIZE != model.config.image_size:
        model.resize_pos_embeddings(model.config.image_size, IMAGE_SIZE, model.config.patch_size)
        model.config.image_size = IMAGE_SIZE
    assert IMAGE_SIZE % model.config.patch_size == 0

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

    output_dir = build_output_dir()
    prepare_output_dir(output_dir)

    shard_index = 0
    shard_file_paths: list[str] = []
    shard_labels: list[int | None] = []
    shard_cls_tokens_batches: list[torch.Tensor] = []
    shard_patch_tokens_batches: list[torch.Tensor] = []
    shard_hidden_states_batches: list[torch.Tensor] = []
    buffered_items = 0

    with torch.inference_mode():
        for images, masks, labels, file_paths in tqdm(dataloader, desc="Processing batches", unit="batch"):
            images, labels, file_paths = pre_process_images(images, masks, labels, file_paths, BACKGROUND)
            inputs = processor(images=images, return_tensors="pt")
            moved_inputs = {}
            for key, value in inputs.items():
                if PIN_MEMORY:
                    value = value.pin_memory()
                if value.is_floating_point():
                    moved_inputs[key] = value.to(model.device, dtype=MODEL_LOAD_DTYPE, non_blocking=PIN_MEMORY)
                else:
                    moved_inputs[key] = value.to(model.device, non_blocking=PIN_MEMORY)
            outputs = model(**moved_inputs, output_hidden_states=SAVE_HIDDEN_STATES)

            cls_tokens = to_storage_feature(outputs.pooler_output)
            assert cls_tokens.shape[0] == len(labels)
            shard_file_paths.extend(file_paths)
            shard_labels.extend(int(label) if label is not None else None for label in labels)
            shard_cls_tokens_batches.append(cls_tokens)

            if SAVE_PATCH_TOKENS:
                patch_tokens = to_storage_feature(outputs.last_hidden_state[:, 1:, :])
                shard_patch_tokens_batches.append(patch_tokens)

            if SAVE_HIDDEN_STATES:
                assert outputs.hidden_states is not None
                shard_hidden_states_batches.append(to_storage_hidden_states(outputs.hidden_states))

            buffered_items += len(file_paths)
            if buffered_items >= SHARD_SIZE:
                shard_index = flush_shard(
                    output_dir=output_dir,
                    shard_index=shard_index,
                    file_paths=shard_file_paths,
                    labels=shard_labels,
                    cls_token=shard_cls_tokens_batches,
                    patch_tokens=shard_patch_tokens_batches,
                    hidden_states=shard_hidden_states_batches,
                    patch_size=model.config.patch_size,
                    image_size=IMAGE_SIZE,
                )
                buffered_items = 0

    flush_shard(
        output_dir=output_dir,
        shard_index=shard_index,
        file_paths=shard_file_paths,
        labels=shard_labels,
        cls_token=shard_cls_tokens_batches,
        patch_tokens=shard_patch_tokens_batches,
        hidden_states=shard_hidden_states_batches,
        patch_size=model.config.patch_size,
        image_size=IMAGE_SIZE,
    )


if __name__ == "__main__":
    main()
