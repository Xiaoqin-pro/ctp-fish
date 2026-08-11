import torch

from models.dle_resnet import DLEResNet18, positive_class_evidence_concentration


def test_dle_spatial_map_shared_classifier_and_identity():
    model = DLEResNet18(16, pretrained=False)
    image = torch.randn(2, 3, 224, 224)
    mask = torch.zeros(2, 1, 224, 224); mask[:, :, 80:160, 80:160] = 1
    output = model(image, mask)
    assert output["feature_map"].shape == (2, 512, 7, 7)
    assert output["global_logits"].shape == output["masked_logits"].shape == (2, 16)
    assert output["positive_evidence"].shape == (2, 16, 7, 7)
    with torch.no_grad():
        mean_cam = output["class_activation"].mean(dim=(2, 3))
        assert torch.allclose(output["global_logits"] - model.classifier.bias, mean_cam, atol=1e-5, rtol=1e-4)


def test_dle_empty_mask_is_finite():
    model = DLEResNet18(16, pretrained=False)
    output = model(torch.randn(2, 3, 224, 224), torch.zeros(2, 1, 224, 224))
    loss, valid = positive_class_evidence_concentration(output, torch.tensor([0, 1]))
    assert not valid.any()
    assert torch.isfinite(loss)
    loss.backward()


def test_dle_localization_backpropagates():
    model = DLEResNet18(16, pretrained=False)
    mask = torch.zeros(2, 1, 224, 224); mask[:, :, 60:170, 60:170] = 1
    output = model(torch.randn(2, 3, 224, 224), mask)
    loss, valid = positive_class_evidence_concentration(output, torch.tensor([0, 1]))
    assert valid.all()
    loss.backward()
    assert model.classifier.weight.grad is not None
    assert any(parameter.grad is not None for parameter in model.layer4.parameters())
