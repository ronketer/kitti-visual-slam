"""I/O for trusted local pipeline pickles, retaining historical payload formats.

Validation checks container shape and required fields, not numerical correctness
or experiment provenance. Pickles must only be loaded from trusted sources.
"""

import os
import hashlib
import json
import pickle
import tempfile
from pathlib import Path


class CheckpointError(ValueError):
    """A checkpoint cannot be used by the requested pipeline stage."""


TRACKING_FIELDS = frozenset({
    "last_frameId", "last_trackId", "trackId_to_frames", "linkId_to_link",
    "frameId_to_lfeature", "frameId_to_trackIds_list",
    "frameId_to_relative_extrinsics", "frameId_to_absolute_extrinsics",
    "prev_frame_links", "leftover_links",
})
WINDOW_FIELDS = frozenset({
    "start_kf", "end_kf", "graph", "initialEstimate", "pose_keys", "point_keys",
    "tracks", "frames", "result", "abs_start_pose", "abs_end_pose",
})
POSE_GRAPH_FIELDS = frozenset({
    "poses_without_loop_closure", "poses_with_loop_closure", "kf_pose_keys",
    "pose_graph", "initial_estimate", "optimized_result", "relative_poses_and_covariances",
})


def validate_payload(payload, kind):
    """Check the existing stage schemas; optional tracking counters stay optional."""
    def require_fields(value, fields):
        if not isinstance(value, dict) or not fields.issubset(value):
            raise CheckpointError(f"Invalid {kind} checkpoint: missing required dictionary fields.")

    if kind == "tracking":
        require_fields(payload, TRACKING_FIELDS)
        if not isinstance(payload["last_frameId"], int) or payload["last_frameId"] < 0:
            raise CheckpointError("Tracking checkpoint contains no frames.")
    elif kind == "ba":
        if not isinstance(payload, list) or not payload:
            raise CheckpointError("BA checkpoint must contain a nonempty list of windows.")
        for window in payload:
            require_fields(window, WINDOW_FIELDS)
    elif kind == "pose-graph":
        require_fields(payload, POSE_GRAPH_FIELDS)
    else:
        raise ValueError(f"Unknown checkpoint kind: {kind}")


def load_checkpoint(path, kind):
    """Reject LFS placeholders before attempting to deserialize a trusted pickle."""
    path = Path(path)
    try:
        with path.open("rb") as stream:
            header = stream.read(128)
            if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise CheckpointError(
                    f"{path} is a Git LFS pointer, not checkpoint data. "
                    "Restore the LFS payload or rebuild into a new artifacts directory."
                )
            stream.seek(0)
            payload = pickle.load(stream)
    except ModuleNotFoundError as exc:
        raise CheckpointError(
            f"Cannot load {path}: required module {exc.name!r} is unavailable. "
            "GTSAM checkpoints require a compatible GTSAM environment."
        ) from exc
    except (OSError, EOFError, pickle.UnpicklingError, ImportError, AttributeError, ValueError) as exc:
        if isinstance(exc, CheckpointError):
            raise
        raise CheckpointError(f"Cannot load checkpoint {path}: {exc}") from exc
    validate_payload(payload, kind)
    return payload


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_exists(path, kind, *, upstream=None):
    """Only a readable, schema-checked checkpoint is eligible for resume."""
    if not Path(path).exists():
        return False
    load_checkpoint(path, kind)
    if upstream is not None:
        try:
            metadata = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
            expected = {"schema": 1, "checkpoint_sha256": file_digest(path),
                        "upstream_sha256": file_digest(upstream)}
            if metadata != expected:
                raise ValueError("checkpoint or upstream digest changed")
        except (OSError, ValueError) as exc:
            raise CheckpointError(
                f"Cannot resume {path}: missing or mismatched dependency metadata. "
                "Use --force to rebuild, or choose a new output directory."
            ) from exc
    return True


def _atomic_write(path, writer):
    """Publish only a completed write; preserve the previous file on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.stem + "-", suffix=".pkl", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path, payload, kind, *, upstream=None):
    validate_payload(payload, kind)

    def write(temporary):
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream)

    _atomic_write(path, write)
    if upstream is not None:
        metadata = {"schema": 1, "checkpoint_sha256": file_digest(path),
                    "upstream_sha256": file_digest(upstream)}
        _atomic_write(str(path) + ".json", lambda temporary: temporary.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"))


def save_tracking_checkpoint(path, database):
    """Use TrackingDB's existing serializer and class paths without duplicating it."""
    def write(temporary):
        database.serialize(str(temporary.with_suffix("")))
        load_checkpoint(temporary, "tracking")

    _atomic_write(path, write)
