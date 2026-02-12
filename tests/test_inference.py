import torch
from src.inference_app import SimpleCNN, predict_tensor

def test_predict_tensor_output():

    model = SimpleCNN()
    model.eval()

    x = torch.randn(1, 3, 224, 224)

    prob, label = predict_tensor(model, x)

    assert isinstance(prob, float)
    assert label in ["cat", "dog"]
