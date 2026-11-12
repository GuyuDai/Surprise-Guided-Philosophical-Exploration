import torch
import pytest

from philo_mcts.decoding import (
    _apply_repetition_penalty,
    annealed_alpha,
    is_degenerate_suffix,
    sparse_contrastive_logits,
)


def test_annealing_reaches_zero_at_exact_token():
    assert annealed_alpha(1.0, 0, 50) == 1.0
    assert annealed_alpha(1.0, 48, 50) == pytest.approx(1 / 49)
    assert annealed_alpha(1.0, 49, 50) == 0.0
    assert annealed_alpha(1.0, 199, 200) == 0.0


def test_only_amateur_top_k_logits_are_subtracted():
    master = torch.tensor([4.0, 3.0, 2.0, 1.0])
    amateur = torch.tensor([8.0, 7.0, 1.0, 0.0])
    combined = sparse_contrastive_logits(master, amateur, alpha=1.0, amateur_top_k=2)
    assert torch.equal(combined, torch.tensor([-4.0, -4.0, 2.0, 1.0]))


def test_repetition_penalty_matches_transformers_sign_rule():
    logits = torch.tensor([4.0, -4.0, 2.0])
    penalized = _apply_repetition_penalty(logits, [0, 1, 1], 2.0)
    assert torch.equal(penalized, torch.tensor([2.0, -8.0, 2.0]))


def test_degenerate_suffix_detector():
    assert is_degenerate_suffix("answer!!!!!!!!")
    assert is_degenerate_suffix("aaaaaaaa")
    assert is_degenerate_suffix("abcabcabcabcabcabc")
    assert is_degenerate_suffix("\ufffd\ufffd\ufffd\ufffd")
    assert not is_degenerate_suffix("### Title\n---\nA structured argument.")
    assert not is_degenerate_suffix("A reasonable answer!   \n")

