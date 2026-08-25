"""High-resolution multi-view tiling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PIL import Image


TilingMode = Literal["none", "grid_2x2"]


@dataclass(frozen=True)
class TileMetadata:
    tile_id: str
    crop_xyxy: tuple[int, int, int, int]
    original_width: int
    original_height: int


@dataclass(frozen=True)
class TiledImage:
    image: Image.Image
    metadata: TileMetadata


def _axis_splits(length: int, overlap_ratio: float) -> tuple[tuple[int, int], tuple[int, int]]:
    overlap = int(round((length / 2) * overlap_ratio))
    mid = length // 2
    first = (0, min(length, mid + overlap))
    second = (max(0, mid - overlap), length)
    return first, second


def make_tiled_views(
    image: Image.Image,
    mode: TilingMode = "none",
    overlap_ratio: float = 0.0,
) -> list[TiledImage]:
    """Return global view plus optional local crops.

    `grid_2x2` returns five images in a fixed order:
    global overview, top-left, top-right, bottom-left, bottom-right.
    """

    if overlap_ratio < 0 or overlap_ratio >= 0.5:
        raise ValueError("overlap_ratio must be in [0, 0.5)")

    source = image.convert("RGB")
    width, height = source.size
    global_view = TiledImage(
        image=source.copy(),
        metadata=TileMetadata(
            tile_id="global",
            crop_xyxy=(0, 0, width, height),
            original_width=width,
            original_height=height,
        ),
    )
    if mode == "none":
        return [global_view]
    if mode != "grid_2x2":
        raise ValueError(f"unsupported tiling mode: {mode}")

    x_ranges = _axis_splits(width, overlap_ratio)
    y_ranges = _axis_splits(height, overlap_ratio)
    tiles = [global_view]
    for tile_id, y_range, x_range in [
        ("top_left", y_ranges[0], x_ranges[0]),
        ("top_right", y_ranges[0], x_ranges[1]),
        ("bottom_left", y_ranges[1], x_ranges[0]),
        ("bottom_right", y_ranges[1], x_ranges[1]),
    ]:
        crop = (x_range[0], y_range[0], x_range[1], y_range[1])
        tiles.append(
            TiledImage(
                image=source.crop(crop),
                metadata=TileMetadata(
                    tile_id=tile_id,
                    crop_xyxy=crop,
                    original_width=width,
                    original_height=height,
                ),
            )
        )
    return tiles


def tiling_prompt_prefix() -> str:
    return (
        "You are given multiple views of the same remote-sensing image.\n"
        "Image 1 is the complete global overview.\n"
        "Image 2 is the top-left high-resolution detail crop of the same image.\n"
        "Image 3 is the top-right high-resolution detail crop of the same image.\n"
        "Image 4 is the bottom-left high-resolution detail crop of the same image.\n"
        "Image 5 is the bottom-right high-resolution detail crop of the same image.\n"
        "Use the local detail crops only to recover fine visual details from the same scene. "
        "Do not treat the five images as five different scenes. Answer the original question.\n"
    )
