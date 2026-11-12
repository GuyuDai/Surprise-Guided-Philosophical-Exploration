from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

import torch
import torch.nn.functional as F
from transformers import StoppingCriteria, StoppingCriteriaList

from .models import input_device


EMERGENCY_MAX_NEW_TOKENS = 4096


@dataclass(frozen=True)
class Generation:
    token_ids: List[int]
    metadata: Dict[str, Any]


class GenerationStrategy(Protocol):
    """Extension point for phase-2 surprise scoring and MCTS."""

    name: str

    def generate(self, prompt_ids: List[int], seed: int) -> Generation:
        ...


def annealed_alpha(initial: float, step: int, zero_at_token: int) -> float:
    """Linearly reach zero exactly at the one-indexed target token."""
    if zero_at_token <= 1:
        return 0.0
    return initial * max(0.0, 1.0 - step / (zero_at_token - 1))


def sparse_contrastive_logits(
    master_logits: torch.Tensor,
    amateur_logits: torch.Tensor,
    alpha: float,
    amateur_top_k: int,
) -> torch.Tensor:
    """Compute master - alpha * amateur, with amateur sparse outside top-k."""
    if master_logits.shape != amateur_logits.shape:
        raise ValueError("master and amateur logits must have the same shape")
    if amateur_top_k <= 0:
        raise ValueError("amateur_top_k must be positive")
    k = min(amateur_top_k, amateur_logits.numel())
    values, indices = torch.topk(amateur_logits.float(), k=k)
    sparse = torch.zeros_like(master_logits, dtype=torch.float)
    sparse.scatter_(0, indices.to(master_logits.device), values.to(master_logits.device))
    return master_logits.float() - alpha * sparse


def is_degenerate_suffix(text: str) -> bool:
    tail = text.rstrip()
    if not tail:
        return False

    if len(tail) >= 8 and all(
        unicodedata.category(char).startswith("P") for char in tail[-8:]
    ):
        return True
    if len(tail) >= 8 and not tail[-1].isspace() and tail.endswith(tail[-1] * 8):
        return True

    for size in range(2, 5):
        length = size * ((16 + size - 1) // size)
        if len(tail) >= length:
            suffix = tail[-length:]
            if suffix == suffix[:size] * (length // size):
                return True

    def invalid(char: str) -> bool:
        return char == "\ufffd" or unicodedata.category(char) in {"Cc", "Cf", "Co", "Cs"}

    return len(text) >= 4 and all(invalid(char) for char in text[-4:])


class DegenerationStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: Any, prompt_length: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **_: Any
    ) -> bool:
        text = self.tokenizer.decode(
            input_ids[0, self.prompt_length :], skip_special_tokens=True
        )
        return is_degenerate_suffix(text)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_repetition_penalty(
    logits: torch.Tensor, token_ids: List[int], penalty: float
) -> torch.Tensor:
    if penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    if penalty == 1.0 or not token_ids:
        return logits
    result = logits.clone()
    indices = torch.tensor(sorted(set(token_ids)), device=result.device)
    selected = result[indices]
    result[indices] = torch.where(selected < 0, selected * penalty, selected / penalty)
    return result


def _sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> int:
    if temperature <= 0:
        raise ValueError("temperature must be positive for sampling")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    scaled = logits.float() / temperature
    k = min(top_k, scaled.numel())
    cutoff = torch.topk(scaled, k=k).values[-1]
    scaled = scaled.masked_fill(scaled < cutoff, -torch.inf)
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
    probabilities = F.softmax(sorted_logits, dim=-1)
    remove = probabilities.cumsum(dim=-1) > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    probabilities = probabilities.masked_fill(remove, 0.0)
    chosen = torch.multinomial(probabilities, 1)
    return int(sorted_indices[chosen].item())


def _eos_ids(model: Any) -> set:
    ids = model.generation_config.eos_token_id
    if ids is None:
        return set()
    return set(ids if isinstance(ids, list) else [ids])


@dataclass
class MasterSampler:
    model: Any
    tokenizer: Any
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    name: str

    @torch.inference_mode()
    def generate(self, prompt_ids: List[int], seed: int) -> Generation:
        _seed_everything(seed)
        device = input_device(self.model)
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(inputs)
        output = self.model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=EMERGENCY_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList(
                [DegenerationStoppingCriteria(self.tokenizer, len(prompt_ids))]
            ),
        )
        new_ids = output[0, inputs.shape[1] :].tolist()
        return Generation(
            new_ids,
            {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
            },
        )


@dataclass
class ContrastiveDecoder:
    master: Any
    amateur: Any
    tokenizer: Any
    vocab_size: int
    alpha: float
    amateur_top_k: int
    zero_at_token: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    name: str

    @torch.inference_mode()
    def generate(self, prompt_ids: List[int], seed: int) -> Generation:
        _seed_everything(seed)
        master_device = input_device(self.master)
        amateur_device = input_device(self.amateur)
        master_input = torch.tensor([prompt_ids], dtype=torch.long, device=master_device)
        amateur_input = master_input.to(amateur_device)
        master_mask = torch.ones_like(master_input)
        amateur_mask = torch.ones_like(amateur_input)
        master_cache = amateur_cache = None
        generated: List[int] = []
        eos_ids = _eos_ids(self.master)

        for step in range(EMERGENCY_MAX_NEW_TOKENS):
            master_out = self.master(
                input_ids=master_input,
                attention_mask=master_mask,
                past_key_values=master_cache,
                use_cache=True,
            )
            master_cache = master_out.past_key_values
            master_logits = master_out.logits[0, -1, : self.vocab_size]
            current_alpha = annealed_alpha(self.alpha, step, self.zero_at_token)

            if current_alpha > 0:
                amateur_out = self.amateur(
                    input_ids=amateur_input,
                    attention_mask=amateur_mask,
                    past_key_values=amateur_cache,
                    use_cache=True,
                )
                amateur_cache = amateur_out.past_key_values
                combined = sparse_contrastive_logits(
                    master_logits,
                    amateur_out.logits[0, -1, : self.vocab_size],
                    current_alpha,
                    self.amateur_top_k,
                )
            else:
                combined = master_logits.float()

            combined = _apply_repetition_penalty(
                combined, prompt_ids + generated, self.repetition_penalty
            )
            token_id = _sample(combined, self.temperature, self.top_p, self.top_k)
            generated.append(token_id)
            if token_id in eos_ids:
                break
            if is_degenerate_suffix(
                self.tokenizer.decode(generated, skip_special_tokens=True)
            ):
                break
            master_input = torch.tensor([[token_id]], device=master_device)
            master_mask = torch.cat(
                [master_mask, torch.ones((1, 1), dtype=torch.long, device=master_device)],
                dim=1,
            )
            if current_alpha > 0:
                amateur_input = torch.tensor([[token_id]], device=amateur_device)
                amateur_mask = torch.cat(
                    [
                        amateur_mask,
                        torch.ones((1, 1), dtype=torch.long, device=amateur_device),
                    ],
                    dim=1,
                )

        return Generation(
            generated,
            {
                "alpha": self.alpha,
                "amateur_top_k": self.amateur_top_k,
                "zero_at_token": self.zero_at_token,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
            },
        )

