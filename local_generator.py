"""Self-contained local Transformers generator used by the subtree mutator."""

from __future__ import annotations

import importlib.util
import os
from typing import Any


class LocalGenerator:
    def __init__(
        self,
        *,
        model_name: str,
        adapter_path: str | None = None,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        batch_size: int = 1,
        gpu_devices: list[int] | None = None,
    ) -> None:
        print("[LLM] Importing transformers and torch...", flush=True)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        requested = [int(item) for item in (gpu_devices or [0])]
        self.device_map = self._device_map(requested)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_options: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if torch.cuda.is_available():
            model_options["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_options["attn_implementation"] = (
                "flash_attention_2" if importlib.util.find_spec("flash_attn") is not None else "sdpa"
            )
        print(f"[LLM] Loading model: {model_name}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_options).eval()

        if adapter_path:
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(f"LoRA adapter does not exist: {adapter_path}")
            from peft import PeftModel

            print(f"[LLM] Loading LoRA adapter: {adapter_path}", flush=True)
            self.model = PeftModel.from_pretrained(self.model, adapter_path).merge_and_unload().eval()

        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = max(1, int(batch_size))

    def _device_map(self, requested: list[int]) -> str | dict[str, int]:
        torch = self.torch
        if not torch.cuda.is_available():
            print("[LLM] CUDA unavailable; using CPU", flush=True)
            return "cpu"
        visible_count = torch.cuda.device_count()
        if any(device < 0 or device >= visible_count for device in requested):
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL_VISIBLE")
            raise ValueError(
                f"requested --gpu {requested}, but torch sees local indices 0..{visible_count - 1} "
                f"with CUDA_VISIBLE_DEVICES={visible}"
            )
        torch.cuda.empty_cache()
        if len(requested) == 1:
            print(f"[LLM] Loading on cuda:{requested[0]}", flush=True)
            return {"": requested[0]}
        print(f"[LLM] Loading across visible CUDA devices {requested}", flush=True)
        return "auto"

    def generate_batch(self, prompts: list[str]) -> list[str | None]:
        results: list[str | None] = []
        for start in range(0, len(prompts), self.batch_size):
            results.extend(self._generate(prompts[start : start + self.batch_size]))
        return results

    def _generate(self, prompts: list[str]) -> list[str | None]:
        if not prompts:
            return []
        torch = self.torch
        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(self.model.device)
            input_length = inputs.input_ids.shape[1]
            generation_options: dict[str, Any] = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            if self.temperature > 0:
                generation_options["temperature"] = self.temperature
            with torch.no_grad():
                output = self.model.generate(**generation_options)
            return self.tokenizer.batch_decode(output[:, input_length:], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(prompts) == 1:
                return [None]
            midpoint = len(prompts) // 2
            return self._generate(prompts[:midpoint]) + self._generate(prompts[midpoint:])

