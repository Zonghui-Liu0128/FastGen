from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image


def read_video_frames(path: str | Path, sequence_length: int) -> torch.Tensor:
    """Read the first sequence_length RGB frames as uint8 [T, H, W, C]."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to read videos. Install requirements.txt first.") from exc

    frames = []
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {path}")
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= sequence_length:
                break

    if len(frames) < sequence_length:
        raise ValueError(f"Video {path} has {len(frames)} decoded frames, expected at least {sequence_length}")
    return torch.from_numpy(__import__("numpy").stack(frames, axis=0))


def transform_video_frames(
    frames: torch.Tensor,
    *,
    sequence_length: int,
    img_size: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Match VideoWDSLoader video transform.

    Args:
        frames: uint8 RGB tensor [T, H, W, C].
        sequence_length: number of frames to keep.
        img_size: target size as (width, height).

    Returns:
        video: float tensor [C, T, H, W] normalized to [-1, 1].
        cropping_params: resize/crop parameters from the center crop.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames shape [T, H, W, C], got {tuple(frames.shape)}")
    if frames.shape[0] < sequence_length:
        raise ValueError(f"Video too short: {frames.shape[0]} < {sequence_length}")

    video = frames[:sequence_length].permute(0, 3, 1, 2)
    video = _resize_small_side_aspect_preserving(video, img_size)
    video, cropping_params = _center_crop(video, img_size, return_cropping_params=True)
    video = video.float() / 127.5 - 1.0
    video = video.permute(1, 0, 2, 3).contiguous()
    return video, cropping_params


def load_first_frame_image(path: str | Path, *, img_size: tuple[int, int]) -> torch.Tensor:
    """Load an explicit conditioning image as [C, 1, H, W] in [-1, 1]."""
    image = Image.open(path).convert("RGB")
    frame = transforms_F.pil_to_tensor(image).unsqueeze(0)
    frame = _resize_small_side_aspect_preserving(frame, img_size)
    frame = _center_crop(frame, img_size)
    frame = frame.float() / 127.5 - 1.0
    return frame.permute(1, 0, 2, 3).contiguous()


def first_frame_from_transformed_video(video: torch.Tensor) -> torch.Tensor:
    """Return the transformed first video frame as [C, 1, H, W]."""
    if video.ndim != 4:
        raise ValueError(f"Expected transformed video [C, T, H, W], got {tuple(video.shape)}")
    return video[:, 0:1].contiguous()


def write_video_preview(path: str | Path, video: torch.Tensor, fps: int = 16) -> None:
    """Write a decoded [-1, 1] video tensor to mp4 for visual inspection."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to write preview videos. Install requirements.txt first.") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError("Preview writer only supports a single decoded video at a time")
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected decoded video [C, T, H, W], got {tuple(video.shape)}")

    frames = video.detach().float().cpu().clamp(-1, 1)
    frames = ((frames + 1.0) * 127.5).round().to(torch.uint8)
    frames = frames.permute(1, 2, 3, 0).numpy()

    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        for frame_array in frames:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _resize_small_side_aspect_preserving(data: torch.Tensor, img_size: tuple[int, int]) -> torch.Tensor:
    img_w, img_h = img_size
    orig_h, orig_w = data.shape[-2:]
    scaling_ratio = max((img_w / orig_w), (img_h / orig_h))
    target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))
    if target_size[0] < img_h or target_size[1] < img_w:
        raise ValueError(f"Resize error. orig {(orig_w, orig_h)} target {img_size} computed {target_size}")
    return transforms_F.resize(
        data,
        size=target_size,
        interpolation=transforms_F.InterpolationMode.BICUBIC,
        antialias=True,
    )


def _center_crop(
    data: torch.Tensor,
    img_size: tuple[int, int],
    return_cropping_params: bool = False,
):
    img_w, img_h = img_size
    orig_h, orig_w = data.shape[-2:]
    cropped = transforms_F.center_crop(data, [img_h, img_w])
    crop_x0 = (orig_w - img_w) // 2
    crop_y0 = (orig_h - img_h) // 2
    cropping_params = {
        "resize_w": orig_w,
        "resize_h": orig_h,
        "crop_x0": crop_x0,
        "crop_y0": crop_y0,
        "crop_w": img_w,
        "crop_h": img_h,
    }
    if return_cropping_params:
        return cropped, cropping_params
    return cropped
