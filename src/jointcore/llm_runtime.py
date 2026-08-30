"""Pinned local Transformers runtime for JointCore stateful actors."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from .utils import sha256_file


class InputBoundError(RuntimeError):
    """Raised before generation when a frozen prompt cap would be exceeded."""


def validate_model_config(payload: dict[str, Any], expected_roles: set[str]) -> None:
    if payload.get("schema_version") != 1 or payload.get("idea_id") != "jointcore":
        raise ValueError("Wrong JointCore local-model configuration")
    runtime = payload.get("runtime", {})
    if runtime.get("serving_engine") != "transformers" or runtime.get("torch_dtype") != "bfloat16":
        raise ValueError("JointCore requires the pinned BF16 Transformers runtime")
    if runtime.get("transformers_expected") != "5.15.0" or runtime.get("decoding") != "greedy":
        raise ValueError("JointCore runtime version or decoding policy changed")
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Local model inventory is absent")
    for key, spec in models.items():
        required = {"model_id", "revision", "path_env", "tokenizer_revision", "role", "quantization", "chat_template", "max_new_tokens"}
        if not required <= set(spec):
            raise ValueError(f"Model {key} lacks required runtime provenance")
        if spec["role"] not in expected_roles:
            raise ValueError(f"Model {key} violates its registered role")
        quantization = str(spec["quantization"])
        if quantization == "none_bfloat16":
            pass
        elif quantization == "bnb_4bit_nf4_double_quant":
            if spec["role"] not in {"posthoc_extension_actor", "paper_expansion_actor"}:
                raise ValueError(f"Model {key} may use 4-bit loading only in a registered extension study")
        else:
            raise ValueError(f"Model {key} has an unsupported quantization contract")
        path_env = str(spec["path_env"])
        if not path_env.startswith("JOINTCORE_") or not path_env.endswith("_PATH"):
            raise ValueError(f"Model {key} has an invalid path environment variable")
        if int(spec["max_new_tokens"]) <= 0:
            raise ValueError(f"Model {key} token cap is invalid")


def model_spec(payload: dict[str, Any], key: str, expected_roles: set[str]) -> dict[str, Any]:
    validate_model_config(payload, expected_roles)
    if key not in payload["models"]:
        raise ValueError(f"Unknown registered JointCore actor: {key}")
    spec = dict(payload["models"][key])
    env_name = str(spec["path_env"])
    configured = os.environ.get(env_name)
    if not configured:
        raise RuntimeError(f"Set {env_name} to the local directory for {spec['model_id']}")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{env_name} must contain an absolute path")
    spec["local_path"] = str(path)
    return spec


def runtime_metadata(spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(spec["local_path"])
    model_config, tokenizer_config = path / "config.json", path / "tokenizer_config.json"
    if not model_config.exists() or not tokenizer_config.exists():
        raise FileNotFoundError(f"Incomplete local snapshot: {path}")
    return {
        "model_id": spec["model_id"],
        "model_revision": spec["revision"],
        "tokenizer_revision": spec["tokenizer_revision"],
        "model_config_sha256": sha256_file(model_config),
        "tokenizer_config_sha256": sha256_file(tokenizer_config),
        "quantization": spec["quantization"],
        "chat_template": spec["chat_template"],
        "fix_mistral_regex": bool(spec.get("fix_mistral_regex", False)),
        "python": sys.version.split()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
    }


def load_tokenizer(spec: dict[str, Any]) -> Any:
    """Load the pinned tokenizer with any registered compatibility fix."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        Path(spec["local_path"]),
        local_files_only=True,
        fix_mistral_regex=bool(spec.get("fix_mistral_regex", False)),
    )


def load_model(spec: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Protected allocation must expose exactly CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one allocation-visible CUDA device")
    path = Path(spec["local_path"])
    metadata = runtime_metadata(spec)
    metadata.update({
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "cuda_device_name": torch.cuda.get_device_name(0),
    })
    started = time.perf_counter()
    tokenizer = load_tokenizer(spec)
    quantization = str(spec["quantization"])
    loader = str(spec.get("model_loader", "auto_causal_lm"))
    if quantization == "none_bfloat16":
        if loader != "auto_causal_lm":
            raise ValueError("Unquantized JointCore actors require the causal-LM loader")
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        model.to("cuda")
    else:
        if loader == "auto_causal_lm":
            loader_class = AutoModelForCausalLM
        elif loader == "auto_image_text_to_text":
            loader_class = AutoModelForImageTextToText
        else:
            raise ValueError(f"Unsupported quantized JointCore model loader: {loader}")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = loader_class.from_pretrained(
            path,
            quantization_config=quantization_config,
            device_map={"": 0},
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        import bitsandbytes

        metadata["bitsandbytes"] = bitsandbytes.__version__
        metadata["quantized_compute_dtype"] = "bfloat16"
        metadata["quantized_type"] = "nf4"
        metadata["quantized_double_quant"] = True
    model.eval()
    torch.cuda.synchronize()
    metadata["model_load_seconds"] = time.perf_counter() - started
    metadata["gpu_memory_allocated_bytes_after_load"] = int(torch.cuda.memory_allocated())
    metadata["gpu_memory_reserved_bytes_after_load"] = int(torch.cuda.memory_reserved())
    metadata["model_loader"] = loader
    return tokenizer, model, metadata


def render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Registered tokenizer lacks a chat template")
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def greedy_generate(
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    max_input_tokens: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    prompt = render_chat(tokenizer, messages)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_tokens = int(encoded["input_ids"].shape[-1])
    if input_tokens > max_input_tokens:
        raise InputBoundError(f"Prompt has {input_tokens} tokens, above frozen bound {max_input_tokens}")
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    completion_ids = generated[0, input_tokens:]
    return {
        "prompt": prompt,
        "completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
        "input_tokens": input_tokens,
        "output_tokens": int(completion_ids.shape[-1]),
        "generation_seconds": elapsed,
        "tokens_per_second": int(completion_ids.shape[-1]) / elapsed if elapsed else None,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "decoding": {"do_sample": False, "max_new_tokens": max_new_tokens, "seed": seed, "use_cache": True},
    }


def score_choices(
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    choices: Sequence[str],
    max_input_tokens: int,
) -> dict[str, Any]:
    """Choose an action by length-normalized conditional log likelihood.

    This is an additive post-hoc adapter.  The frozen free-generation runtime
    above is intentionally unchanged.  Candidate tokens are appended to the
    rendered assistant prefix explicitly so that formatting cannot turn a
    valid semantic choice into a parser failure.
    """
    import torch

    if not choices or len(set(choices)) != len(choices):
        raise ValueError("Choice scoring requires distinct non-empty choices")
    prompt = render_chat(tokenizer, messages)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not prompt_ids:
        raise ValueError("Rendered chat prompt is empty")
    encoded_choices: list[tuple[str, list[int]]] = []
    for choice in choices:
        choice_ids = tokenizer(str(choice), add_special_tokens=False)["input_ids"]
        if not choice_ids:
            raise ValueError(f"Choice tokenization is empty: {choice!r}")
        combined_length = len(prompt_ids) + len(choice_ids)
        if combined_length > max_input_tokens:
            raise InputBoundError(
                f"Prompt plus choice has {combined_length} tokens, above post-hoc bound {max_input_tokens}"
            )
        encoded_choices.append((str(choice), choice_ids))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Choice-scoring tokenizer has neither pad nor EOS token")
    maximum_length = max(len(prompt_ids) + len(choice_ids) for _, choice_ids in encoded_choices)
    batch_ids: list[list[int]] = []
    batch_masks: list[list[int]] = []
    for _, choice_ids in encoded_choices:
        combined = prompt_ids + choice_ids
        padding = maximum_length - len(combined)
        batch_ids.append(combined + [int(pad_id)] * padding)
        batch_masks.append([1] * len(combined) + [0] * padding)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        input_ids = torch.tensor(batch_ids, dtype=torch.long, device="cuda")
        attention_mask = torch.tensor(batch_masks, dtype=torch.long, device="cuda")
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        scored: list[dict[str, Any]] = []
        for row, (choice, choice_ids) in enumerate(encoded_choices):
            start = len(prompt_ids) - 1
            stop = start + len(choice_ids)
            token_logits = logits[row, start:stop].float()
            targets = torch.tensor(choice_ids, dtype=torch.long, device="cuda")
            token_log_probs = torch.log_softmax(token_logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
            total = float(token_log_probs.sum().item())
            scored.append({
                "choice": str(choice),
                "token_ids": [int(item) for item in choice_ids],
                "token_count": len(choice_ids),
                "total_log_probability": total,
                "mean_log_probability": total / len(choice_ids),
            })
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    ranked = sorted(scored, key=lambda item: (-float(item["mean_log_probability"]), choices.index(str(item["choice"]))))
    selected = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    return {
        "prompt": prompt,
        "completion": selected["choice"],
        "choice_scores": scored,
        "choice_scoring": "mean_conditional_log_probability",
        "choice_margin": (
            float(selected["mean_log_probability"]) - float(runner_up["mean_log_probability"])
            if runner_up is not None else None
        ),
        "input_tokens": len(prompt_ids),
        "output_tokens": int(selected["token_count"]),
        "generation_seconds": elapsed,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "decoding": {"method": "constrained_choice_scoring", "choices": list(choices), "use_cache": False},
    }
