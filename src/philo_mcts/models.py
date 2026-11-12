from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelConfig:
    master: str
    amateur: str
    dtype: str = "bfloat16"
    device_map: str = "auto"


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    payload = {
        "vocab": sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]),
        "special_tokens": {k: str(v) for k, v in tokenizer.special_tokens_map.items()},
        "chat_template": tokenizer.chat_template,
        "backend": tokenizer.backend_tokenizer.to_str(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_shared_tokenizer(master_id: str, amateur_id: str) -> Tuple[Any, str]:
    """Load and prove that both checkpoints map every token to the same id."""
    master = AutoTokenizer.from_pretrained(master_id, use_fast=True)
    amateur = AutoTokenizer.from_pretrained(amateur_id, use_fast=True)
    master_hash = _tokenizer_fingerprint(master)
    amateur_hash = _tokenizer_fingerprint(amateur)
    if master_hash != amateur_hash:
        raise ValueError(
            "Master and amateur tokenizers differ; contrastive decoding is unsafe: "
            f"{master_hash[:12]} != {amateur_hash[:12]}"
        )
    return master, master_hash


def _dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def load_models(config: ModelConfig, tokenizer_size: int) -> Tuple[Any, Any]:
    common = {
        "dtype": _dtype(config.dtype),
        "device_map": config.device_map,
        "low_cpu_mem_usage": True,
    }
    master = AutoModelForCausalLM.from_pretrained(config.master, **common).eval()
    amateur = AutoModelForCausalLM.from_pretrained(config.amateur, **common).eval()
    for name, model in (("master", master), ("amateur", amateur)):
        if model.get_input_embeddings().num_embeddings < tokenizer_size:
            raise ValueError(f"{name} embedding table does not cover the shared tokenizer")
    return master, amateur


def input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def chat_token_ids(tokenizer: Any, messages: List[Dict[str, str]]) -> List[int]:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]

