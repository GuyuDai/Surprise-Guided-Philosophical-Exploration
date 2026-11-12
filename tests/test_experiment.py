import json

import pytest

from philo_mcts import experiment


def test_prompt_reader_accepts_jsonl_and_multiline_objects(tmp_path):
    records = [
        {"id": "one", "prompt": '中文问题：如何理解 "{真}"？'},
        {"id": "two", "prompt": "第一行\n第二行"},
    ]
    path = tmp_path / "prompts.jsonl"
    for indent in (None, 4):
        path.write_text(
            "\n\n".join(json.dumps(item, ensure_ascii=False, indent=indent) for item in records),
            encoding="utf-8",
        )
        assert list(experiment._read_prompts(path)) == records


@pytest.mark.parametrize("invalid", ['{"id":', '{"id":"bad","prompt":42}', ''])
def test_invalid_prompts_fail_before_model_loading(tmp_path, monkeypatch, invalid):
    path = tmp_path / "prompts.jsonl"
    content = '{"id":"valid","prompt":"question"}\n' + invalid if invalid else ''
    path.write_text(content, encoding="utf-8")

    def unexpected_load(*args):
        pytest.fail("model loading must not run before prompt validation")

    monkeypatch.setattr(experiment, "load_shared_tokenizer", unexpected_load)
    config = {"prompts": str(path), "models": {"master": "unused", "amateur": "unused"}}
    with pytest.raises(ValueError, match=str(path)):
        experiment.run(config, tmp_path / "configs" / "phase1.json")


def test_markdown_report_contains_every_answer(tmp_path):
    payload = {
        "created_at": "2026-11-12T00:00:00+00:00",
        "models": {"master": "master", "amateur": "amateur"},
        "tokenizer_sha256": "digest",
        "experiment": {
            "generation": {
                "seeds": [1],
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.05,
                "high_temperature": 1.5,
                "high_top_p": 0.95,
            },
            "contrastive": {
                "alpha": 1.0,
                "amateur_top_k": 50,
                "fast_zero_at_token": 50,
                "slow_zero_at_token": 200,
            },
            "system_prompt": "",
        },
        "prompts": [
            {
                "id": "question",
                "prompt": "Why?",
                "outputs": {
                    strategy: [
                        {
                            "seed": 1,
                            "response": f"answer from {strategy}",
                            "generated_token_count": 3,
                            "elapsed_seconds": 0.1,
                            "metadata": {"temperature": 0.7},
                        }
                    ]
                    for strategy in (
                        "master_normal",
                        "master_high_temperature",
                        "contrastive_alpha_zero_at_50",
                        "contrastive_alpha_zero_at_200",
                    )
                },
            }
        ],
    }
    path = tmp_path / "phase1.md"
    experiment._write_markdown(path, payload)
    report = path.read_text(encoding="utf-8")
    assert "完成度：4/4 个答案（完整）" in report
    assert "System prompt：无" in report
    assert report.count("#### Seed 1") == 4
