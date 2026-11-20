"""Small generated media tests; no private data or model downloads."""

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("scenedetect")
pytest.importorskip("av")


def test_known_hard_cut_and_original_timeline(tmp_path):
    from beauty_video_templates.contracts import validate_result
    from beauty_video_templates.parsing.backends import shots

    video = tmp_path / "cut.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
    if not writer.isOpened():
        pytest.skip("MJPG encoder unavailable")
    for color in [(0, 0, 255), (255, 0, 0)]:
        for _ in range(30):
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            image[:] = color
            writer.write(image)
    writer.release()
    result = shots(video)
    validate_result(result)
    assert len(result["items"]) == 2
    assert result["items"][1]["start_frame"] == 30
    assert result["items"][1]["start"] == pytest.approx(1.0, abs=1 / 30)
    assert result["items"][-1]["end"] == pytest.approx(2.0, abs=1 / 30)
    from beauty_video_templates.parsing.media import frame, prepare_frames

    prepare_frames(video, [0.5, 1.5])
    assert frame(video, 0.5)[0, 0, 2] > 200
    assert frame(video, 1.5)[0, 0, 0] > 200
