import pytest

torch = pytest.importorskip("torch")

from src.models.cnn import SmallCNN
from src.models.resnet import build_resnet18

BATCH = 2
NUM_CLASSES = 38


@pytest.mark.parametrize("num_classes", [2, 38])
def test_cnn_accepts_batch_and_outputs_logits(num_classes):
    model = SmallCNN(num_classes=num_classes)
    out = model(torch.randn(BATCH, 3, 224, 224))
    assert out.shape == (BATCH, num_classes)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("num_classes", [2, 38])
def test_resnet_accepts_batch_and_outputs_logits(num_classes):
    model = build_resnet18(num_classes, pretrained=False)
    out = model(torch.randn(BATCH, 3, 224, 224))
    assert out.shape == (BATCH, num_classes)
    assert out.dtype == torch.float32


def test_models_output_raw_logits_without_softmax():
    models = [SmallCNN(NUM_CLASSES), build_resnet18(NUM_CLASSES, pretrained=False)]
    for model in models:
        for module in model.modules():
            assert not isinstance(module, (torch.nn.Softmax, torch.nn.LogSoftmax)), type(module)


def test_resnet_last_layer_is_linear():
    model = build_resnet18(NUM_CLASSES, pretrained=False)
    assert isinstance(model.fc, torch.nn.Linear)
    assert model.fc.out_features == NUM_CLASSES


def test_resnet_freeze_backbone_only_fc_trainable():
    model = build_resnet18(NUM_CLASSES, pretrained=False, freeze_backbone=True, finetune_layer4=False)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["fc.weight", "fc.bias"]


def test_resnet_finetune_layer4_unfreezes_layer4():
    model = build_resnet18(NUM_CLASSES, pretrained=False, freeze_backbone=True, finetune_layer4=True)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    assert {"fc.weight", "fc.bias"}.issubset(trainable)
    assert any(name.startswith("layer4.") for name in trainable)
    assert not any(name.startswith("layer4.") for name in frozen)
    assert any(name.startswith("conv1.") for name in frozen)
    assert any(name.startswith("layer1.") for name in frozen)


def test_resnet_no_freeze_all_trainable():
    model = build_resnet18(NUM_CLASSES, pretrained=False, freeze_backbone=False)
    assert all(p.requires_grad for p in model.parameters())
