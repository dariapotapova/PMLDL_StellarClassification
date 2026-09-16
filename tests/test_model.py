"""Architecture tests for the numerical tabular classifier.

References
----------
Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A.
"Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021.
https://arxiv.org/abs/2106.11959

PyTorch TransformerEncoderLayer documentation:
https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html

PyTorch TransformerEncoder documentation:
https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html

PyTorch BatchNorm1d documentation:
https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

PyTorch Linear documentation:
https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
"""

import pytest
import torch

from src.model import (
    FTTransformerClassifier,
    HybridFTTransformerClassifier,
    TabularResNetClassifier,
)


@pytest.mark.parametrize("batch_size", [1, 4, 11])
def test_output_shape_for_different_batch_sizes(batch_size: int) -> None:
    model = FTTransformerClassifier()
    inputs = torch.randn(batch_size, 14)

    logits = model(inputs)

    assert logits.shape == (batch_size, 3)


def test_invalid_number_of_features_raises_value_error() -> None:
    model = FTTransformerClassifier()

    with pytest.raises(ValueError, match="Expected 14 features"):
        model(torch.randn(4, 13))


@pytest.mark.parametrize(
    "invalid_input",
    [torch.randn(14), torch.randn(2, 3, 14)],
)
def test_invalid_tensor_dimensionality_raises_value_error(
    invalid_input: torch.Tensor,
) -> None:
    model = FTTransformerClassifier()

    with pytest.raises(ValueError, match="Expected a 2D input tensor"):
        model(invalid_input)


def test_outputs_are_finite_for_finite_input() -> None:
    model = FTTransformerClassifier()
    inputs = torch.randn(8, 14)

    logits = model(inputs)

    assert torch.isfinite(inputs).all()
    assert torch.isfinite(logits).all()


def test_backward_propagation_produces_parameter_gradients() -> None:
    model = FTTransformerClassifier()
    inputs = torch.randn(5, 14)

    model(inputs).sum().backward()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert trainable_parameters
    assert all(parameter.grad is not None for parameter in trainable_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable_parameters)


def test_eval_mode_is_deterministic_for_same_input() -> None:
    model = FTTransformerClassifier().eval()
    inputs = torch.randn(6, 14)

    with torch.no_grad():
        first_logits = model(inputs)
        second_logits = model(inputs)

    assert torch.equal(first_logits, second_logits)


def test_hybrid_output_shape() -> None:
    model = HybridFTTransformerClassifier()

    logits = model(torch.randn(4, 14))

    assert logits.shape == (4, 3)


def test_hybrid_outputs_are_finite() -> None:
    model = HybridFTTransformerClassifier()

    logits = model(torch.randn(8, 14))

    assert torch.isfinite(logits).all()


@pytest.mark.parametrize(
    "invalid_input",
    [torch.randn(13), torch.randn(2, 13), torch.randn(2, 3, 14)],
)
def test_hybrid_invalid_input_raises_value_error(
    invalid_input: torch.Tensor,
) -> None:
    model = HybridFTTransformerClassifier()

    with pytest.raises(ValueError):
        model(invalid_input)


def test_hybrid_backward_propagation_produces_parameter_gradients() -> None:
    model = HybridFTTransformerClassifier()

    model(torch.randn(5, 14)).sum().backward()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert trainable_parameters
    assert all(parameter.grad is not None for parameter in trainable_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable_parameters)


def test_hybrid_eval_mode_is_deterministic() -> None:
    model = HybridFTTransformerClassifier().eval()
    inputs = torch.randn(6, 14)

    with torch.no_grad():
        first_logits = model(inputs)
        second_logits = model(inputs)

    assert torch.equal(first_logits, second_logits)


def test_tabular_resnet_output_shape() -> None:
    model = TabularResNetClassifier()

    logits = model(torch.randn(32, 14))

    assert logits.shape == (32, 3)


@pytest.mark.parametrize("batch_size", [2, 7, 19])
def test_tabular_resnet_supports_different_batch_sizes(batch_size: int) -> None:
    model = TabularResNetClassifier()

    logits = model(torch.randn(batch_size, 14))

    assert logits.shape == (batch_size, 3)


def test_tabular_resnet_incorrect_feature_count_raises_value_error() -> None:
    model = TabularResNetClassifier()

    with pytest.raises(ValueError, match="Expected 14 features"):
        model(torch.randn(4, 13))


@pytest.mark.parametrize("invalid_input", [torch.randn(14), torch.randn(2, 3, 14)])
def test_tabular_resnet_incorrect_dimensionality_raises_value_error(
    invalid_input: torch.Tensor,
) -> None:
    model = TabularResNetClassifier()

    with pytest.raises(ValueError, match="Expected a 2D input tensor"):
        model(invalid_input)


def test_tabular_resnet_outputs_are_finite() -> None:
    model = TabularResNetClassifier()

    logits = model(torch.randn(8, 14))

    assert torch.isfinite(logits).all()


def test_tabular_resnet_backward_propagation_produces_parameter_gradients() -> None:
    model = TabularResNetClassifier()

    model(torch.randn(5, 14)).sum().backward()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert trainable_parameters
    assert all(parameter.grad is not None for parameter in trainable_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable_parameters)


@pytest.mark.parametrize(
    ("parameter_name", "invalid_value"),
    [
        ("num_features", 0),
        ("num_classes", 0),
        ("hidden_dim", 0),
        ("block_hidden_dim", 0),
        ("num_blocks", 0),
        ("dropout", -0.1),
        ("dropout", 1.0),
    ],
)
def test_tabular_resnet_rejects_invalid_constructor_values(
    parameter_name: str,
    invalid_value: int | float,
) -> None:
    with pytest.raises(ValueError):
        TabularResNetClassifier(**{parameter_name: invalid_value})


def test_tabular_resnet_eval_mode_is_deterministic() -> None:
    model = TabularResNetClassifier().eval()
    inputs = torch.randn(6, 14)

    with torch.no_grad():
        first_logits = model(inputs)
        second_logits = model(inputs)

    assert torch.equal(first_logits, second_logits)
