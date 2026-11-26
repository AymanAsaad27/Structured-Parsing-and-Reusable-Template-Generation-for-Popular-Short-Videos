"""Small-model inference helpers using local, explicitly supplied checkpoints."""


def transition_model(weights):
    import torch
    from torch import nn

    torch.set_num_threads(2)
    raw = torch.load(weights, map_location="cpu", weights_only=True)
    state = raw.get("state_dict", raw)
    classes = raw.get("labels", ["hard_cut", "non_hard_cut"])
    state = {key.removeprefix("layers."): value for key, value in state.items()}
    output_count = state["7.weight"].shape[0]
    if len(classes) != output_count:
        raise ValueError("Checkpoint labels do not match classifier outputs")
    model = nn.Sequential(
        nn.Conv2d(6, 12, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(12, 24, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(24, output_count),
    )
    model.load_state_dict(state)
    model.class_labels = classes
    return model.eval()
