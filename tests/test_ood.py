import pytest

torch = pytest.importorskip("torch")

from src.training.ood import energy_score, msp


def test_msp_equals_max_probability():
    logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 0.0, 0.0]])
    probs = torch.softmax(logits, dim=1)
    values = msp(probs)
    assert values.shape == (2,)
    assert torch.allclose(values, probs.max(dim=1).values)


def test_energy_score_is_lower_than_negative_max_logit():
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    energy = energy_score(logits)
    assert energy.shape == (1,)
    assert energy[0].item() < -2.0
