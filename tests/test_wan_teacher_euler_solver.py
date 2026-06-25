# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from pathlib import Path
import sys
import types

import torch


class _FakeDictConfig(dict):
    def __init__(self, content=None, flags=None, **kwargs):
        super().__init__(content or {}, **kwargs)
        self.__dict__ = self


class _FakeListConfig(list):
    def __init__(self, content=None, flags=None):
        super().__init__(content or [])


class _DummyPretrained:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    @classmethod
    def load_config(cls, *args, **kwargs):
        return {}

    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls()


class _FakeNoiseScheduler:
    min_t = 0.0
    max_t = 0.999

    def __init__(self):
        self.t_init = None

    def safe_clamp(self, t, min=None, max=None):
        return torch.clamp(t, min=min, max=max)

    def latents(self, noise, t_init=None):
        self.t_init = t_init
        return noise.clone()


class _FakeUniPCScheduler:
    def __init__(self):
        self.config = types.SimpleNamespace(num_train_timesteps=1000, flow_shift=None)
        self.step_calls = []

    def set_timesteps(self, num_inference_steps, device):
        self.timesteps = torch.linspace(999, 0, num_inference_steps, device=device)

    def step(self, model_output, timestep, sample, return_dict=False):
        self.step_calls.append((model_output.detach().clone(), timestep.detach().clone()))
        return (sample + 100.0,)


class _FakeEulerScheduler:
    instances = []

    def __init__(self, shift=1.0):
        self.shift = shift
        self.config = types.SimpleNamespace(num_train_timesteps=1000)
        self.step_calls = []
        _FakeEulerScheduler.instances.append(self)

    def set_timesteps(self, num_inference_steps, device):
        self.timesteps = torch.linspace(999, 0, num_inference_steps, device=device)

    def step(self, model_output, timestep, sample, return_dict=False):
        self.step_calls.append((model_output.detach().clone(), timestep.detach().clone()))
        return (sample + model_output,)


def _install_dependency_stubs(monkeypatch):
    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", lambda module, **kwargs: module, raising=False)

    omegaconf = types.ModuleType("omegaconf")
    omegaconf.DictConfig = _FakeDictConfig
    omegaconf.ListConfig = _FakeListConfig

    scipy = types.ModuleType("scipy")
    scipy_stats = types.ModuleType("scipy.stats")
    scipy.stats = scipy_stats

    diffusers = types.ModuleType("diffusers")
    diffusers.DDIMScheduler = _DummyPretrained
    diffusers.CogVideoXDPMScheduler = _DummyPretrained
    diffusers.UniPCMultistepScheduler = _DummyPretrained
    diffusers.FlowMatchEulerDiscreteScheduler = _FakeEulerScheduler

    diffusers_models = types.ModuleType("diffusers.models")
    diffusers_models.WanTransformer3DModel = _DummyPretrained
    diffusers_models.AutoencoderKLWan = _DummyPretrained

    diffusers_transformers = types.ModuleType("diffusers.models.transformers")
    transformer_wan = types.ModuleType("diffusers.models.transformers.transformer_wan")
    transformer_wan.WanTransformerBlock = type("WanTransformerBlock", (), {})
    transformer_wan.WanRotaryPosEmbed = type("WanRotaryPosEmbed", (), {})

    diffusers_utils = types.ModuleType("diffusers.utils")
    diffusers_utils.USE_PEFT_BACKEND = False
    diffusers_utils.scale_lora_layers = lambda *args, **kwargs: None
    diffusers_utils.unscale_lora_layers = lambda *args, **kwargs: None

    diffusers_image_processor = types.ModuleType("diffusers.image_processor")
    diffusers_image_processor.PipelineImageInput = object

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = _DummyPretrained
    transformers.UMT5EncoderModel = _DummyPretrained
    transformers.CLIPImageProcessor = _DummyPretrained
    transformers.CLIPVisionModel = _DummyPretrained

    configs_net = types.ModuleType("fastgen.configs.net")
    configs_net.EDM_CIFAR10_Config = _FakeDictConfig()
    configs_callbacks = types.ModuleType("fastgen.configs.callbacks")
    configs_callbacks.WANDB_CALLBACK = _FakeDictConfig()
    configs_data = types.ModuleType("fastgen.configs.data")
    configs_data.CIFAR10_Loader_Config = _FakeDictConfig()
    configs_opt = types.ModuleType("fastgen.configs.opt")
    configs_opt.BaseOptimizerConfig = _FakeDictConfig()
    configs_opt.BaseSchedulerConfig = _FakeDictConfig()
    configs_opt.get_scheduler = lambda *args, **kwargs: None
    methods = types.ModuleType("fastgen.methods")
    methods.FastGenModel = type("FastGenModel", (), {})
    basic_utils = types.ModuleType("fastgen.utils.basic_utils")
    basic_utils.PRECISION_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    basic_utils.prompt_clean = lambda text: text
    basic_utils.str2bool = lambda value: str(value).lower() in ("true", "1", "yes")
    distributed_fsdp = types.ModuleType("fastgen.utils.distributed.fsdp")
    distributed_fsdp.apply_fsdp_checkpointing = lambda *args, **kwargs: None

    for name, module in {
        "omegaconf": omegaconf,
        "scipy": scipy,
        "scipy.stats": scipy_stats,
        "diffusers": diffusers,
        "diffusers.models": diffusers_models,
        "diffusers.models.transformers": diffusers_transformers,
        "diffusers.models.transformers.transformer_wan": transformer_wan,
        "diffusers.utils": diffusers_utils,
        "diffusers.image_processor": diffusers_image_processor,
        "transformers": transformers,
        "fastgen.configs.net": configs_net,
        "fastgen.configs.callbacks": configs_callbacks,
        "fastgen.configs.data": configs_data,
        "fastgen.configs.opt": configs_opt,
        "fastgen.methods": methods,
        "fastgen.utils.basic_utils": basic_utils,
        "fastgen.utils.distributed.fsdp": distributed_fsdp,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    for name in [
        "fastgen.configs.config",
        "fastgen.networks.Wan.network",
        "fastgen.networks.WanI2V.network",
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_base_model_config_defaults_teacher_solver_to_unipc(monkeypatch):
    _install_dependency_stubs(monkeypatch)
    config_module = importlib.import_module("fastgen.configs.config")

    assert config_module.BaseModelConfig().teacher_solver == "unipc"


def test_video_inference_exposes_teacher_solver_cli_override():
    source = Path("scripts/inference/video_model_inference.py").read_text()

    assert '"--teacher_solver"' in source
    assert 'choices=["unipc", "euler"]' in source
    assert 'args.teacher_solver or getattr(config.model, "teacher_solver", "unipc")' in source


def test_wan_teacher_sample_uses_flowmatch_euler_when_requested(monkeypatch):
    _install_dependency_stubs(monkeypatch)
    wan_module = importlib.import_module("fastgen.networks.Wan.network")
    _FakeEulerScheduler.instances.clear()

    class FakeWan(wan_module.Wan):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.schedule_type = "rf"
            self.noise_scheduler = _FakeNoiseScheduler()
            self._unipc_scheduler = _FakeUniPCScheduler()
            self.forward_calls = []

        def forward(self, latents, t, condition=None, fwd_pred_type=None, **kwargs):
            self.forward_calls.append({"condition": condition, "fwd_pred_type": fwd_pred_type})
            return torch.ones_like(latents) if condition == "neg" else torch.full_like(latents, 3.0)

    model = FakeWan()
    noise = torch.zeros(1, 1, 2, 1, 1)

    model.sample(noise, condition="pos", neg_condition="neg", guidance_scale=2.0, num_steps=3, shift=7.0, solver="euler")

    assert len(_FakeEulerScheduler.instances) == 1
    assert _FakeEulerScheduler.instances[0].shift == 7.0
    assert model._unipc_scheduler.step_calls == []
    assert {call["fwd_pred_type"] for call in model.forward_calls} == {"flow"}
    for model_output, _ in _FakeEulerScheduler.instances[0].step_calls:
        assert torch.allclose(model_output, torch.full_like(model_output, 5.0))


def test_wani2v_teacher_euler_preserves_first_frame_condition(monkeypatch):
    _install_dependency_stubs(monkeypatch)
    wani2v_module = importlib.import_module("fastgen.networks.WanI2V.network")
    _FakeEulerScheduler.instances.clear()

    class FakeWanI2V(wani2v_module.WanI2V):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.schedule_type = "rf"
            self.concat_mask = False
            self.noise_scheduler = _FakeNoiseScheduler()
            self._unipc_scheduler = _FakeUniPCScheduler()

        def forward(self, latents, t, condition=None, fwd_pred_type=None, **kwargs):
            return torch.ones_like(latents) if condition["text_embeds"] == "neg" else torch.full_like(latents, 3.0)

    model = FakeWanI2V()
    noise = torch.zeros(1, 1, 3, 1, 1)
    first_frame_cond = torch.full_like(noise, -2.0)
    condition = {"text_embeds": "pos", "first_frame_cond": first_frame_cond}
    neg_condition = {"text_embeds": "neg", "first_frame_cond": first_frame_cond}

    output = model.sample(
        noise,
        condition=condition,
        neg_condition=neg_condition,
        guidance_scale=2.0,
        num_steps=3,
        shift=7.0,
        solver="euler",
    )

    assert len(_FakeEulerScheduler.instances) == 1
    assert model._unipc_scheduler.step_calls == []
    for model_output, _ in _FakeEulerScheduler.instances[0].step_calls:
        assert torch.allclose(model_output[:, :, 0], torch.zeros_like(model_output[:, :, 0]))
    assert torch.allclose(output[:, :, 0], first_frame_cond[:, :, 0])
