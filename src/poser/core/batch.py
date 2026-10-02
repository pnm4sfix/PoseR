"""Run pose estimation or behaviour decoding over many files."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .behaviour_decode import decode_behaviours
from .io import save_coords_to_h5
from .pose_estimation import PoseEstimator
from .schemas.batch import BatchMode, BatchResult
from .schemas.pose_estimation import InferenceMode
from .schemas.training import TrainingConfig
from .settings import resolve_device

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["pose_file", "video_file", "output", "status", "error"]


class BatchJob(BaseModel):
    """Configuration for a multi-file batch analysis run.

    Built from CLI arguments and from user code, so the fields are validated
    on construction rather than part way through a long run. Call run() to
    execute it; the work itself lives in BatchRunner.

    Attributes:
        video_files: Paired with pose_files by position. Shorter lists are
            padded with empty strings.
        mode: "behaviour" detects and classifies bouts in each pose file;
            "pose_estimation" runs YOLO-pose over each video.
        checkpoint: Model to run. A YOLO pose model under "pose_estimation",
            an ST-GCN classifier under "behaviour".
        config: A TrainingConfig, or a path to a config.yaml.
        output_dir: Where per-file outputs and batch_manifest.csv are written.
        n_individuals: Upper bound on detections per frame, passed to YOLO as
            max_det.
        progress_callback: Called as (completed, total, pose_path) after each
            file. Exceptions it raises are logged, not propagated.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pose_files: List[str] = Field(default_factory=list)
    video_files: List[str] = Field(default_factory=list)
    mode: BatchMode = BatchMode.BEHAVIOUR
    checkpoint: str = ""
    config: Optional[TrainingConfig] = None
    output_dir: str = ""
    n_individuals: int = 1
    progress_callback: Optional[Callable[[int, int, str], None]] = None

    @field_validator("pose_files", "video_files", mode="before")
    @classmethod
    def _stringify_path_list(cls, value):
        """Accept Path entries, which both callers build with pathlib."""
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return value

    @field_validator("checkpoint", "output_dir", mode="before")
    @classmethod
    def _stringify_path(cls, value):
        """Accept None and Path where a plain string is expected."""
        return "" if value is None else str(value)

    @field_validator("config", mode="before")
    @classmethod
    def _load_config(cls, value):
        """Accept a path to a config.yaml as well as a TrainingConfig."""
        if isinstance(value, (str, Path)):
            return TrainingConfig.from_yaml(value)
        return value

    def run(self) -> List[BatchResult]:
        """Execute this job. See BatchRunner.run."""
        return BatchRunner(self).run()


class BatchRunner:
    """Executes a BatchJob: iterates inputs, captures failures, writes a manifest."""

    def __init__(self, job: BatchJob):
        self._job = job

    def run(self) -> List[BatchResult]:
        """Process every input and write batch_manifest.csv beside the outputs.

        A file that fails is recorded as an error result and does not stop the
        rest of the run.

        Returns:
            One BatchResult per input, in input order.
        """
        job = self._job
        Path(job.output_dir or ".").mkdir(parents=True, exist_ok=True)

        pairs = self._input_pairs()
        total = len(pairs)
        results: List[BatchResult] = []

        for pose_path, video_path in pairs:
            results.append(self._process_one(pose_path, video_path))
            self._report(len(results), total, pose_path)

        self._write_manifest(results)
        return results

    def _input_pairs(self) -> List[Tuple[str, str]]:
        """Pair each input with its counterpart, padding the shorter list.

        Behaviour decoding is driven by pose_files, pose estimation by
        video_files. Pose estimation falls back to pose_files when no videos
        were given, because the CLI's only positional argument is pose_files.
        """
        job = self._job
        poses = list(job.pose_files)
        videos = list(job.video_files)

        if job.mode is BatchMode.POSE_ESTIMATION:
            videos = videos or poses
            poses += [""] * (len(videos) - len(poses))
        else:
            videos += [""] * (len(poses) - len(videos))

        return list(zip(poses, videos))

    def _process_one(self, pose_path: str, video_path: str) -> BatchResult:
        """Run one input, returning an error result rather than raising."""
        job = self._job
        try:
            if job.mode is BatchMode.POSE_ESTIMATION:
                estimator = PoseEstimator()
                # Open the video first, so a missing one fails before a model loads.
                estimator.input_load(video_path)
                estimator.model_load(
                    job.checkpoint or "yolo11n-pose.pt", resolve_device()
                )
                records = list(
                    estimator.iter_keypoints(
                        InferenceMode.TRACK, max_individuals=job.n_individuals
                    )
                )
                coords = estimator.coords_from_keypoints(records)
                output = save_coords_to_h5(coords, video_path)
            else:
                output = decode_behaviours(
                    pose_path, job.checkpoint, job.config, job.output_dir
                )
        except Exception as exc:
            log.error("Error processing %s: %s", pose_path, exc)
            return BatchResult(
                pose_path=pose_path,
                video_path=video_path,
                output_path="",
                status="error",
                error=str(exc),
            )

        return BatchResult(
            pose_path=pose_path,
            video_path=video_path,
            output_path=output,
            status="ok",
        )

    def _report(self, completed: int, total: int, pose_path: str) -> None:
        """Notify the progress callback, if one was given."""
        if self._job.progress_callback is None:
            return
        try:
            self._job.progress_callback(completed, total, pose_path)
        except Exception as exc:
            # A broken callback must not discard work already done.
            log.warning("progress_callback raised: %s", exc)

    def _write_manifest(self, results: List[BatchResult]) -> None:
        """Write one CSV row per result, unless no output_dir was given."""
        if not self._job.output_dir:
            return
        path = Path(self._job.output_dir) / "batch_manifest.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "pose_file": r.pose_path,
                        "video_file": r.video_path,
                        "output": r.output_path,
                        "status": r.status,
                        "error": r.error,
                    }
                )
        log.info("Manifest written to %s", path)
