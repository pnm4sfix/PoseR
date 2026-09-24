"""Run pose estimation or behaviour decoding over many files."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .behaviour_decode import decode_behaviours
from .pose_estimation import estimate_poses_from_video
from .schemas.batch import BatchMode, BatchResult
from .schemas.training import TrainingConfig

log = logging.getLogger(__name__)


class BatchJob(BaseModel):
    """Configuration for a multi-file batch analysis run.

    Built from CLI arguments and from user code, so the fields are validated
    on construction rather than part way through a long run.

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
        """Process every input and write batch_manifest.csv beside the outputs.

        A file that fails is recorded as an error result and does not stop the
        rest of the run.

        Returns:
            One BatchResult per input, in input order.
        """
        Path(self.output_dir or ".").mkdir(parents=True, exist_ok=True)
        results: List[BatchResult] = []

        pairs = self._input_pairs()
        total = len(pairs)

        for pose_path, video_path in pairs:
            try:
                if self.mode is BatchMode.POSE_ESTIMATION:
                    out = estimate_poses_from_video(
                        video_path, self.checkpoint, self.n_individuals
                    )
                else:
                    out = decode_behaviours(
                        pose_path,
                        self.checkpoint,
                        self.config,
                        self.output_dir,
                    )

                results.append(
                    BatchResult(
                        pose_path=pose_path,
                        video_path=video_path,
                        output_path=out,
                        status="ok",
                    )
                )
            except Exception as exc:
                results.append(
                    BatchResult(
                        pose_path=pose_path,
                        video_path=video_path,
                        output_path="",
                        status="error",
                        error=str(exc),
                    )
                )
                log.error("Error processing %s: %s", pose_path, exc)

            if self.progress_callback:
                try:
                    self.progress_callback(len(results), total, pose_path)
                except Exception as exc:
                    # A broken callback must not discard work already done.
                    log.warning("progress_callback raised: %s", exc)

        self._write_manifest(results)
        return results

    def _input_pairs(self) -> List[tuple[str, str]]:
        """Pair each input with its counterpart, padding the shorter list.

        Behaviour decoding is driven by pose_files, pose estimation by
        video_files. Pose estimation falls back to pose_files when no videos
        were given, because the CLI's only positional argument is pose_files.
        """
        poses = list(self.pose_files)
        videos = list(self.video_files)

        if self.mode is BatchMode.POSE_ESTIMATION:
            videos = videos or poses
            poses += [""] * (len(videos) - len(poses))
        else:
            videos += [""] * (len(poses) - len(videos))

        return list(zip(poses, videos))

    def _write_manifest(self, results: List[BatchResult]) -> None:
        if not self.output_dir:
            return
        path = Path(self.output_dir) / "batch_manifest.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["pose_file", "video_file", "output", "status", "error"]
            )
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
