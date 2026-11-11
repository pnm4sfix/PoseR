"""Read a video file as a lazy array that decodes frames only when indexed."""

from __future__ import annotations

from pathlib import Path

import av
import dask
import dask.array as da
import imageio.v3 as iio
import numpy as np

CHUNK_FRAMES = 64


def read_video(path: str | Path) -> da.Array:
    """Open a video without decoding it.

    Frames are decoded in chunks of CHUNK_FRAMES when the array is indexed, so
    only the chunks being looked at are ever in memory.

    Returns:
        (T, H, W, 3) RGB frames, or (T, H, W) when the first frame is grey.

    Raises:
        OSError: If the file does not report how many frames it holds.
    """
    path = str(path)
    props = iio.improps(path, plugin="pyav")
    n_frames = props.n_images
    if not n_frames:
        raise OSError("Could not read video metadata")
    # props.shape is (H, W, C) or (N, H, W, C), so count from the end.
    height, width = int(props.shape[-3]), int(props.shape[-2])

    first = iio.imread(path, index=0, plugin="pyav")
    grey = (
        first.ndim == 3
        and first.shape[2] == 3
        and np.array_equal(first[:, :, 0], first[:, :, 1])
    )

    chunks = []
    for start in range(0, n_frames, CHUNK_FRAMES):
        count = min(CHUNK_FRAMES, n_frames - start)
        shape = (count, height, width) if grey else (count, height, width, 3)
        chunk = dask.delayed(_read_chunk)(
            path, start, count, height, width, grey, props.dtype
        )
        chunks.append(da.from_delayed(chunk, shape=shape, dtype=props.dtype))
    return da.concatenate(chunks, axis=0)


def _read_chunk(
    path: str,
    start: int,
    count: int,
    height: int,
    width: int,
    grey: bool,
    dtype: np.dtype,
) -> np.ndarray:
    """Decode count frames from start: seek to the keyframe before, read forward."""
    shape = (count, height, width) if grey else (count, height, width, 3)
    out = np.empty(shape, dtype=dtype)
    written = 0
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 25.0)
        time_base = float(stream.time_base)
        if start > 0:
            target = int(start / fps / time_base)
            container.seek(target, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            # The frame's index, from its timestamp: a seek lands on a keyframe
            # at or before start, so the frames before start are skipped.
            if frame.pts is not None:
                index = int(round(float(frame.pts or 0) * time_base * fps))
            else:
                index = start + written
            if index < start:
                continue
            if written >= count:
                break
            image = frame.to_ndarray(format="rgb24")
            out[written] = image[:, :, 0] if grey else image
            written += 1
    # A file that ends early leaves the rest of the chunk unwritten.
    return out
