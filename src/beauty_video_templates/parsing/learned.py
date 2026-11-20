"""Small-model inference helpers using local, explicitly supplied checkpoints."""


def transition_model(weights):
    import torch
    from torch import nn

    model = nn.Sequential(
        nn.Conv2d(6, 12, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(12, 24, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(24, 2),
    )
    raw = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("layers."): value for key, value in raw.items()})
    return model.eval()
