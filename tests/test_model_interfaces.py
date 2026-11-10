import pytest
torch = pytest.importorskip("torch")
from src.models.cnn import SmallCNN
from src.models.resnet import build_resnet18


def test_cnn_shape():
    model = SmallCNN(num_classes=4)
    assert model(torch.randn(2, 3, 224, 224)).shape == (2, 4)


def test_resnet_shape_without_download():
    model = build_resnet18(4, pretrained=False)
    assert model(torch.randn(2, 3, 224, 224)).shape == (2, 4)
