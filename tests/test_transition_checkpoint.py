"""Both legacy binary and labelled multiclass checkpoints remain executable."""

import pytest

torch = pytest.importorskip("torch")

from beauty_video_templates.parsing.learned import transition_model  # noqa: E402


@pytest.mark.parametrize(
    "labels",
    [["hard_cut", "non_hard_cut"], ["hard_cut", "jump_cut", "dissolve", "black_transition"]],
)
def test_transition_checkpoint_dimensions_and_labels(tmp_path, labels):
    nn = torch.nn
    model = nn.Sequential(
        nn.Conv2d(6, 12, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(12, 24, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(24, len(labels)),
    )
    state = {"layers." + k: v for k, v in model.state_dict().items()}
    payload = state if len(labels) == 2 else {"state_dict": state, "labels": labels}
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    loaded = transition_model(path)
    assert loaded.class_labels == labels
    with torch.inference_mode():
        assert loaded(torch.zeros(1, 6, 32, 32)).shape == (1, len(labels))
