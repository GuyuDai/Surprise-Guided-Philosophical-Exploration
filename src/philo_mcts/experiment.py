from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from .decoding import ContrastiveDecoder, GenerationStrategy, MasterSampler
from .models import ModelConfig, chat_token_ids, load_models, load_shared_tokenizer


def _read_prompts(path: Path) -> Iterable[Dict[str, str]]:
    # Accept JSONL and whitespace-separated, pretty-printed JSON objects.
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        line_number = text.count("\n", 0, position) + 1
        try:
            item, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in ("id", "prompt")
        ):
            raise ValueError(f"{path}:{line_number} requires non-empty string id and prompt")
        yield item


def _prompt_ids(tokenizer: Any, system_prompt: str, prompt: str) -> List[int]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return chat_token_ids(tokenizer, messages)


def _strategies(config: Dict[str, Any], master: Any, amateur: Any, tokenizer: Any):
    generation = config["generation"]
    contrastive = config["contrastive"]
    sampling = {
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "top_k": generation["top_k"],
        "repetition_penalty": generation["repetition_penalty"],
    }
    contrastive_shared = {
        "master": master,
        "amateur": amateur,
        "tokenizer": tokenizer,
        "vocab_size": len(tokenizer),
        "alpha": contrastive["alpha"],
        "amateur_top_k": contrastive["amateur_top_k"],
        **sampling,
    }
    return [
        MasterSampler(
            model=master,
            tokenizer=tokenizer,
            name="master_normal",
            **sampling,
        ),
        MasterSampler(
            model=master,
            tokenizer=tokenizer,
            temperature=generation["high_temperature"],
            top_p=generation["high_top_p"],
            top_k=generation["top_k"],
            repetition_penalty=generation["repetition_penalty"],
            name="master_high_temperature",
        ),
        ContrastiveDecoder(
            **contrastive_shared,
            zero_at_token=contrastive["fast_zero_at_token"],
            name="contrastive_alpha_zero_at_50",
        ),
        ContrastiveDecoder(
            **contrastive_shared,
            zero_at_token=contrastive["slow_zero_at_token"],
            name="contrastive_alpha_zero_at_200",
        ),
    ]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    strategy_names = {
        "master_normal": "1. Master 默认采样",
        "master_high_temperature": "2. Master 高温采样",
        "contrastive_alpha_zero_at_50": (
            "3. Contrastive decoding，alpha 在第 50 个 token 归零"
        ),
        "contrastive_alpha_zero_at_200": (
            "4. Contrastive decoding，alpha 在第 200 个 token 归零"
        ),
    }

    def quote(text: str) -> str:
        return "\n".join(
            f"> {line}" if line else ">" for line in text.splitlines()
        ) or ">"

    generation = payload["experiment"]["generation"]
    contrastive = payload["experiment"]["contrastive"]
    actual = sum(
        len(answers)
        for prompt in payload["prompts"]
        for answers in prompt["outputs"].values()
    )
    expected = (
        len(payload["prompts"])
        * len(strategy_names)
        * len(generation["seeds"])
    )
    completion = "完整" if actual == expected else "未完成"
    lines = [
        "# Phase 1 实验结果",
        "",
        f"- 完成度：{actual}/{expected} 个答案（{completion}）",
        f"- Prompt 数：{len(payload['prompts'])}",
        f"- 生成时间：{payload['created_at']}",
        f"- Master：{payload['models']['master']}",
        f"- Amateur：{payload['models']['amateur']}",
        f"- Tokenizer SHA-256：`{payload['tokenizer_sha256']}`",
        f"- 源数据：`{path.with_suffix('.json').name}`",
        "",
        "## 实验配置",
        "",
        f"- Seeds：{generation['seeds']}",
        f"- 默认采样：temperature={generation['temperature']}, "
        f"top_p={generation['top_p']}, top_k={generation['top_k']}, "
        f"repetition_penalty={generation['repetition_penalty']}",
        f"- 高温采样：temperature={generation['high_temperature']}, "
        f"top_p={generation['high_top_p']}, top_k={generation['top_k']}, "
        f"repetition_penalty={generation['repetition_penalty']}",
        f"- Contrastive decoding：初始 alpha={contrastive['alpha']}, "
        f"amateur top-k={contrastive['amateur_top_k']}, "
        f"归零位置={contrastive['fast_zero_at_token']}/"
        f"{contrastive['slow_zero_at_token']}",
        f"- System prompt：{payload['experiment'].get('system_prompt') or '无'}",
    ]
    for index, prompt in enumerate(payload["prompts"], 1):
        lines.extend(
            [
                "",
                f"## {index}. {prompt['id']}",
                "",
                "**Prompt**",
                "",
                quote(prompt["prompt"]),
            ]
        )
        for strategy, title in strategy_names.items():
            lines.extend(["", f"### {title}"])
            for answer in prompt["outputs"][strategy]:
                metadata = ", ".join(
                    f"{name}={value}"
                    for name, value in answer["metadata"].items()
                )
                lines.extend(
                    [
                        "",
                        f"#### Seed {answer['seed']}",
                        "",
                        f"生成 token：{answer['generated_token_count']}；"
                        f"耗时：{answer['elapsed_seconds']} 秒；参数：{metadata}",
                        "",
                        quote(answer["response"]),
                    ]
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: Dict[str, Any], config_path: Path) -> Path:
    root = config_path.parent.parent
    prompt_path = root / config["prompts"]
    prompts = list(_read_prompts(prompt_path))
    if not prompts:
        raise ValueError(f"{prompt_path}: no prompts found")
    print(f"validated {len(prompts)} prompts", flush=True)
    model_config = ModelConfig(**config["models"])
    tokenizer, tokenizer_hash = load_shared_tokenizer(
        model_config.master, model_config.amateur
    )
    print(f"tokenizer verified: sha256={tokenizer_hash}", flush=True)
    master, amateur = load_models(model_config, len(tokenizer))
    strategies: List[GenerationStrategy] = _strategies(
        config, master, amateur, tokenizer
    )

    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"phase1-{stamp}.json"
    seeds = config["generation"]["seeds"]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": config["models"],
        "tokenizer_sha256": tokenizer_hash,
        "experiment": {
            "generation": config["generation"],
            "contrastive": config["contrastive"],
            "system_prompt": config.get("system_prompt", ""),
        },
        "prompts": [],
    }
    _write_json(output_path, payload)

    for item in prompts:
        prompt_record = {
            "id": item["id"],
            "prompt": item["prompt"],
            "outputs": {strategy.name: [] for strategy in strategies},
        }
        payload["prompts"].append(prompt_record)
        _write_json(output_path, payload)
        prompt_ids = _prompt_ids(
            tokenizer, config.get("system_prompt", ""), item["prompt"]
        )
        for strategy in strategies:
            for seed in seeds:
                started = time.perf_counter()
                result = strategy.generate(prompt_ids, seed)
                record = {
                    "seed": seed,
                    "response": tokenizer.decode(
                        result.token_ids, skip_special_tokens=True
                    ),
                    "generated_token_count": len(result.token_ids),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "metadata": result.metadata,
                }
                prompt_record["outputs"][strategy.name].append(record)
                _write_json(output_path, payload)
                print(
                    f"[{item['id']}] {strategy.name} seed={seed}: "
                    f"{len(result.token_ids)} tokens",
                    flush=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    markdown_path = output_path.with_suffix(".md")
    _write_markdown(markdown_path, payload)
    print(f"markdown: {markdown_path}", flush=True)
    return output_path

