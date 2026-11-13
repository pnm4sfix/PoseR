"""Tests for poser.core.video."""

from __future__ import annotations

import av
import numpy as np

from poser.core.video import CHUNK_FRAMES, read_video

HEIGHT = 32
WIDTH = 48
# Three chunks, the last one short, so reads cross two chunk boundaries.
N_FRAMES = 2 * CHUNK_FRAMES + 22


def _write_video(path, frames: np.ndarray) -> None:
    """Encode (T, H, W, 3) RGB frames as MPEG-4."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        for image in frames:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _decode_front_to_back(path) -> np.ndarray:
    """Every frame as (T, H, W, 3), decoded in file order with no seeking."""
    with av.open(str(path)) as container:
        return np.stack(
            [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        )


def _brightness(n_frames: int) -> np.ndarray:
    """One level per frame, far enough apart that no two frames look alike."""
    return ((np.arange(n_frames) * 37) % 256).astype(np.uint8)


def test_colour_video_matches_a_front_to_back_decode(tmp_path):
    frames = np.empty((N_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)
    level = _brightness(N_FRAMES)[:, None, None]
    frames[..., 0] = level
    frames[..., 1] = 255 - level
    frames[..., 2] = 128
    path = tmp_path / "colour.avi"
    _write_video(path, frames)

    video = read_video(path)

    assert video.shape == (N_FRAMES, HEIGHT, WIDTH, 3)
    assert video.dtype == np.uint8
    # Each chunk seeks to a keyframe and decodes forward from there, so a
    # wrong seek shows up as frames out of place.
    np.testing.assert_array_equal(np.asarray(video), _decode_front_to_back(path))


def test_grey_video_has_no_channel_axis(tmp_path):
    level = _brightness(N_FRAMES)[:, None, None, None]
    frames = np.broadcast_to(level, (N_FRAMES, HEIGHT, WIDTH, 3))
    path = tmp_path / "grey.avi"
    _write_video(path, np.ascontiguousarray(frames))

    video = read_video(path)

    assert video.shape == (N_FRAMES, HEIGHT, WIDTH)
    np.testing.assert_array_equal(
        np.asarray(video), _decode_front_to_back(path)[..., 0]
    )
