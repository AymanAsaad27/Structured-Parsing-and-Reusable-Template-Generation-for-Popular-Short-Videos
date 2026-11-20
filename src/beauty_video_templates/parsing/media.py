"""Local media utilities; no upload of source material."""

import subprocess
import tempfile
from pathlib import Path

_cached_path = None
_cached_frames = {}


def prepare_indexed_frames(path, indices):
    """Cache selected source-frame indices and return every frame's PTS."""
    global _cached_path, _cached_frames
    import av
    import cv2

    wanted = {int(i) for i in indices}
    _cached_path = str(path)
    _cached_frames = {}
    times = []
    with av.open(str(path)) as container:
        origin = None
        for i, decoded in enumerate(container.decode(video=0)):
            if origin is None:
                origin = float(decoded.time or 0)
            t = float(decoded.time or 0) - origin
            times.append(t)
            if i in wanted:
                image = decoded.to_ndarray(format="bgr24")
                if image.shape[1] > 640:
                    image = cv2.resize(image, (640, round(image.shape[0] * 640 / image.shape[1])))
                _cached_frames[round(t, 6)] = image
    return times


def prepare_frames(path, times):
    """Decode requested presentation-time samples in one pass, bounded to one video."""
    global _cached_path, _cached_frames
    import av
    import cv2

    _cached_path = str(path)
    _cached_frames = {}
    pending = sorted({round(max(0, float(t)), 6) for t in times})
    index = 0
    image = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        for decoded in container.decode(stream):
            if decoded.time is None:
                continue
            time = float(decoded.time) - origin
            if index < len(pending) and time + 1e-6 >= pending[index]:
                image = decoded.to_ndarray(format="bgr24")
                if image.shape[1] > 640:
                    image = cv2.resize(image, (640, round(image.shape[0] * 640 / image.shape[1])))
                while index < len(pending) and time + 1e-6 >= pending[index]:
                    _cached_frames[pending[index]] = image
                    index += 1
            if index == len(pending):
                break
    if index < len(pending) and image is not None:
        for t in pending[index:]:
            _cached_frames[t] = image


def metadata(path):
    import cv2

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps <= 0 or count <= 0:
            raise ValueError(f"Cannot decode video: {path}")
        return {"id": path.stem, "duration_seconds": count / fps}, fps, count
    finally:
        cap.release()


def envelope(path, module, backend, parameters, items):
    from beauty_video_templates.contracts import validate_result

    video, _, _ = metadata(path)
    result = dict(
        schema_version="0.1.0",
        module=module,
        backend=backend,
        video=video,
        parameters=parameters,
        items=items,
    )
    validate_result(result)
    return result


def presentation_times(path):
    """Decode presentation timestamps; preserve variable-frame-rate timing."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        times = [float(f.time) for f in container.decode(stream) if f.time is not None]
        duration = float(stream.duration * stream.time_base) if stream.duration else None
    if not times:
        raise ValueError("Video has no timestamped frames")
    offset = times[0]
    times = [t - offset for t in times]
    tail = (times[-1] - times[-2]) if len(times) > 1 else 1 / metadata(path)[1]
    return times, max(times[-1] + tail, (duration or 0))


def frame(path, seconds, width=640):
    import av
    import cv2

    if str(path) == _cached_path and round(max(0, float(seconds)), 6) in _cached_frames:
        image = _cached_frames[round(max(0, float(seconds)), 6)]
        if image.shape[1] > width:
            image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])))
        return image

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        target = max(0, seconds) + origin
        container.seek(int(target / stream.time_base), stream=stream, backward=True)
        chosen = None
        for decoded in container.decode(stream):
            chosen = decoded
            if decoded.time is not None and float(decoded.time) >= target:
                break
        if chosen is None:
            raise ValueError(f"Cannot read frame at {seconds}")
        image = chosen.to_ndarray(format="bgr24")
        if image.shape[1] > width:
            image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])))
        return image


def audio(path, sample_rate=22050):
    import imageio_ffmpeg
    import soundfile as sf

    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "audio.wav"
        run = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                str(target),
            ],
            capture_output=True,
        )
        if run.returncode:
            message = run.stderr.decode(errors="replace")
            if "does not contain any stream" in message or "matches no streams" in message:
                return None, sample_rate
            raise RuntimeError(message)
        samples, rate = sf.read(target)
    return samples, rate
