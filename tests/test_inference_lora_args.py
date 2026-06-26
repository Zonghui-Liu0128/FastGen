from argparse import Namespace

from scripts.inference.lora_utils import LORA_ADAPTER_NAME, load_lora_adapter


class FakeTransformer:
    def __init__(self, accepts_prefixed=True):
        self.accepts_prefixed = accepts_prefixed
        self.load_calls = []
        self.set_calls = []
        self.peft_config = {}

    def load_lora_adapter(self, source, **kwargs):
        self.load_calls.append((source, kwargs))
        if kwargs.get("prefix") == "transformer" and not self.accepts_prefixed:
            return
        self.peft_config[kwargs["adapter_name"]] = object()

    def set_adapters(self, adapter_name, weights=None):
        self.set_calls.append((adapter_name, weights))


class FakeModule:
    def __init__(self, transformer):
        self.transformer = transformer
        self.to_calls = []
        self.eval_called = False
        self.requires_grad_calls = []

    def to(self, **ctx):
        self.to_calls.append(ctx)
        return self

    def eval(self):
        self.eval_called = True
        return self

    def requires_grad_(self, value):
        self.requires_grad_calls.append(value)
        return self


def test_lora_path_file_is_loaded_with_inferred_weight_name(tmp_path):
    lora_file = tmp_path / "adapter.safetensors"
    lora_file.touch()
    transformer = FakeTransformer()
    module = FakeModule(transformer)
    args = Namespace(lora_path=str(lora_file), lora_scale=0.75)

    load_lora_adapter(module, args, {"device": "cuda", "dtype": "bfloat16"})

    assert transformer.load_calls == [
        (
            str(tmp_path),
            {
                "adapter_name": LORA_ADAPTER_NAME,
                "prefix": "transformer",
                "weight_name": "adapter.safetensors",
            },
        )
    ]
    assert transformer.set_calls == [(LORA_ADAPTER_NAME, 0.75)]
    assert module.to_calls == [{"device": "cuda", "dtype": "bfloat16"}]
    assert module.eval_called
    assert module.requires_grad_calls == [False]


def test_lora_loader_retries_without_prefix_for_transformer_saved_adapters(tmp_path):
    lora_dir = tmp_path / "lora"
    lora_dir.mkdir()
    transformer = FakeTransformer(accepts_prefixed=False)
    module = FakeModule(transformer)
    args = Namespace(lora_path=str(lora_dir), lora_scale=1.0)

    load_lora_adapter(module, args, {})

    assert transformer.load_calls == [
        (str(lora_dir), {"adapter_name": LORA_ADAPTER_NAME, "prefix": "transformer"}),
        (str(lora_dir), {"adapter_name": LORA_ADAPTER_NAME, "prefix": None}),
    ]
    assert transformer.set_calls == [(LORA_ADAPTER_NAME, 1.0)]
