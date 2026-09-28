"""Compatibility entry point for the packaged pipeline."""
import sys
from kitti_slam import pipeline

if __name__ == "__main__":
    raise SystemExit(pipeline.main())
else:
    sys.modules[__name__] = pipeline
