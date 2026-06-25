import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class EncodedSample:
    latent: torch.Tensor
    text_embedding: torch.Tensor
    first_frame_cond: torch.Tensor | None


class WanLatentExtractor:
    """Load Wan VAE/text encoder and reproduce Trainer.preprocess_data offline."""

    def __init__(
        self,
        *,
        model_id_or_local_path: str,
        device: str | torch.device,
        save_dtype: torch.dtype,
        text_precision: torch.dtype = torch.float32,
    ):
        os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        from fastgen.networks.Wan.network import WanTextEncoder, WanVideoEncoder

        self.device = torch.device(device)
        self.save_dtype = save_dtype
        self.text_precision = text_precision
        self.vae = WanVideoEncoder(model_id_or_local_path=model_id_or_local_path).to(self.device)
        self.text_encoder = WanTextEncoder(model_id_or_local_path=model_id_or_local_path).to(self.device)

    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        embedding = self.text_encoder.encode([prompt], precision=self.text_precision)
        return embedding.squeeze(0).to(dtype=self.save_dtype).cpu()

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor, *, mode: str = "sample") -> torch.Tensor:
        latent = self.vae.encode(video.unsqueeze(0).to(self.device), mode=mode)
        return latent.squeeze(0).to(dtype=self.save_dtype).cpu()

    @torch.no_grad()
    def extract(
        self,
        *,
        video: torch.Tensor,
        prompt: str,
        first_frame: torch.Tensor | None = None,
        video_vae_mode: str = "sample",
    ) -> EncodedSample:
        latent = self.encode_video(video, mode=video_vae_mode)
        text_embedding = self.encode_text(prompt)
        first_frame_cond = None
        if first_frame is not None:
            first_frame_cond = self.encode_video(first_frame, mode="argmax")
        return EncodedSample(
            latent=latent,
            text_embedding=text_embedding,
            first_frame_cond=first_frame_cond,
        )

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.vae.decode(latent.unsqueeze(0).to(self.device))
        return decoded.squeeze(0).detach().cpu()


def parse_torch_dtype(name: str) -> torch.dtype:
    dtypes = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return dtypes[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {sorted(dtypes)}") from exc
