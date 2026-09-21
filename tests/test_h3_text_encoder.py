"""The retained H3 state must stay raw, not become the final normalized state."""

from types import SimpleNamespace

import pytest

from vflash.adapters.h3_text_encoder import retain_h3_text_encoder_prefix
from vflash.contracts import ContractError


def encoder(depth=64):
    torch = pytest.importorskip("torch")
    config = SimpleNamespace(num_hidden_layers=depth)
    language = SimpleNamespace(
        layers=torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(depth)]),
        config=config,
    )
    return SimpleNamespace(
        model=SimpleNamespace(language_model=language, visual=object()),
        config=SimpleNamespace(text_config=config),
    )


def states(model, value):
    torch = pytest.importorskip("torch")
    output = []
    for layer in model.model.language_model.layers:
        output.append(value)
        value = value + layer(value).tanh()
    # Mirrors Transformers' hidden-state boundary, including final norm.
    output.append(torch.nn.functional.layer_norm(value, (3,)))
    return output


def test_h3_text_prefix_preserves_raw_state_and_module_identity():
    torch = pytest.importorskip("torch")
    torch.manual_seed(14)
    model = encoder()
    layers = tuple(model.model.language_model.layers)
    visual = model.model.visual
    value = torch.randn(2, 3)
    expected = states(model, value)[50]
    retain_h3_text_encoder_prefix(model)
    assert len(model.model.language_model.layers) == 51
    assert model.config.text_config.num_hidden_layers == 51
    assert model.model.language_model.config.num_hidden_layers == 51
    assert all(
        a is b for a, b in zip(layers[:51], model.model.language_model.layers, strict=True)
    )
    assert model.model.visual is visual
    assert torch.equal(states(model, value)[50], expected)
    # Demonstrate the tempting off-by-one would change the consumed tensor.
    model.model.language_model.layers = model.model.language_model.layers[:50]
    assert not torch.equal(states(model, value)[50], expected)


def test_h3_text_prefix_is_idempotent():
    model = encoder(51)
    layers = tuple(model.model.language_model.layers)
    retain_h3_text_encoder_prefix(model)
    retain_h3_text_encoder_prefix(model)
    assert tuple(model.model.language_model.layers) == layers


@pytest.mark.parametrize("invalid", ["short", "outer", "inner"])
def test_h3_text_prefix_rejects_incoherent_encoder_before_mutation(invalid):
    model = encoder(50 if invalid == "short" else 64)
    language = model.model.language_model
    if invalid == "outer":
        model.config.text_config = SimpleNamespace(num_hidden_layers=63)
    elif invalid == "inner":
        language.config = SimpleNamespace(num_hidden_layers=63)
    original = language.layers
    with pytest.raises(ContractError, match="at least 51 layers"):
        retain_h3_text_encoder_prefix(model)
    assert language.layers is original
