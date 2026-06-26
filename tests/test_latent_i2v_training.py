# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import importlib.machinery
import types
from types import SimpleNamespace

import torch


class _FakeWandbMedia:
    def __init__(self, *args, **kwargs):
        pass


_fake_wandb_util = types.ModuleType("wandb.util")
_fake_wandb_util.generate_id = lambda: "unit-test"
_fake_wandb_util.__spec__ = importlib.machinery.ModuleSpec("wandb.util", loader=None)
_fake_wandb = types.ModuleType("wandb")
_fake_wandb.Image = _FakeWandbMedia
_fake_wandb.Video = _FakeWandbMedia
_fake_wandb.init = lambda *args, **kwargs: None
_fake_wandb.log = lambda *args, **kwargs: None
_fake_wandb.run = None
_fake_wandb.util = _fake_wandb_util
_fake_wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
sys.modules.setdefault("wandb", _fake_wandb)
sys.modules.setdefault("wandb.util", _fake_wandb_util)
_fake_boto3 = types.ModuleType("boto3")
_fake_boto3.client = lambda *args, **kwargs: None
_fake_boto3.__spec__ = importlib.machinery.ModuleSpec("boto3", loader=None)
sys.modules.setdefault("boto3", _fake_boto3)
_fake_av = types.ModuleType("av")
_fake_av.open = lambda *args, **kwargs: None
_fake_av.__spec__ = importlib.machinery.ModuleSpec("av", loader=None)
sys.modules.setdefault("av", _fake_av)

from fastgen.callbacks.wandb import WandbCallback
from fastgen.trainer import Trainer


class _DummyI2VNet:
    is_i2v = True
    concat_mask = False


class _DummyLatentModel:
    device = torch.device("cpu")
    precision = torch.float32
    precision_amp_enc = None
    input_shape = [48, 3, 2, 2]
    net = _DummyI2VNet()


def test_preprocess_data_preserves_loaded_latent_i2v_first_frame_condition():
    trainer = Trainer.__new__(Trainer)
    real = torch.zeros(1, 48, 3, 2, 2)
    loaded_first_frame = torch.full((1, 48, 1, 2, 2), 7.0)
    data = {
        "real": real,
        "condition": torch.ones(1, 512, 4),
        "neg_condition": torch.zeros(1, 512, 4),
        "first_frame_cond": loaded_first_frame.clone(),
    }

    processed = trainer.preprocess_data(_DummyLatentModel(), data)

    assert torch.equal(processed["first_frame_cond"], loaded_first_frame)


def test_wandb_log_media_false_skips_sample_generation_and_vae_initialization():
    callback = WandbCallback(log_media=False)
    callback.config = SimpleNamespace(trainer=SimpleNamespace(logging_iter=1), log_config=SimpleNamespace())
    callback.sample_logging_iter = 1

    class _Net:
        def init_vae(self):
            raise AssertionError("VAE should not be initialized when media logging is disabled")

    model = SimpleNamespace(
        scheduler_dict={},
        net=_Net(),
        device=torch.device("cpu"),
        precision=torch.float32,
        precision_amp_enc=None,
    )

    def generate_sample():
        raise AssertionError("sample callable should not run when media logging is disabled")

    callback.on_training_step_end(
        model,
        data_batch={"real": torch.zeros(1, 48, 3, 2, 2)},
        output_batch={"gen_rand": generate_sample},
        loss_dict={"loss": torch.tensor(1.0)},
        iteration=1,
    )
