from utility import find_camera_location
from tracking_database import TrackingDB
from typing import Optional, Tuple, Dict, List
import numpy as np
from dataclasses import dataclass, field

DEFAULT_MIN_BUNDLE_SIZE = 5
DEFAULT_MAX_BUNDLE_SIZE = 20
DEFAULT_MIN_FEATURES = 50
DEFAULT_MAX_FEATURES = 2000
DEFAULT_MIN_OVERLAP = 0.15
DEFAULT_MIN_TRANSLATION = 0.3
DEFAULT_MAX_TRANSLATION = 14.0


class KeyframeCriterion:
    """Base class for keyframe selection criteria"""
    def evaluate(self, start_frame_id: int, possible_end_frame_id: int, db: TrackingDB) -> bool:
        """
        Evaluate if a possible end frame meets this criterion
        
        Args:
            start_frame_id: Starting frame of the window
            possible_end_frame_id: Potential end frame being evaluated
            db: TrackingDB instance containing tracking information
            
        Returns:
            bool: True if criterion is met for the possible end frame, False otherwise
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

class FeatureCountCriterion(KeyframeCriterion):
    """Criterion checking if frame has appropriate number of tracked features"""
    def __init__(self, min_features: int, max_features: int):
        self.min_features = min_features
        self.max_features = max_features
        
    def evaluate(self, start_frame_id: int, possible_end_frame_id: int, db: TrackingDB) -> bool:
        tracked_features = len(db.tracks(possible_end_frame_id))
        return self.min_features <= tracked_features <= self.max_features

class TrackOverlapCriterion(KeyframeCriterion):
    """Criterion checking track continuity between consecutive frames"""
    def __init__(self, min_overlap: float):
        self.min_overlap = min_overlap
        
    def evaluate(self, start_frame_id: int, possible_end_frame_id: int, db: TrackingDB) -> bool:
        if possible_end_frame_id == 0:
            return False
            
        curr_tracks = set(db.tracks(start_frame_id))
        prev_tracks = set(db.tracks(possible_end_frame_id - 1))
        
        if not curr_tracks or not prev_tracks:
            return False
            
        overlap_ratio = len(curr_tracks & prev_tracks) / len(prev_tracks)
        return overlap_ratio >= self.min_overlap

class TranslationCriterion(KeyframeCriterion):
    """Criterion checking minimum translation from last keyframe"""
    def __init__(self, min_translation: float, max_translation: float):
        self.min_translation = min_translation
        self.max_translation = max_translation
        
    def evaluate(self, start_frame_id: int, possible_end_frame_id: int, db: TrackingDB) -> bool:
        curr_pose = find_camera_location(db.get_absolute_extrinsics(possible_end_frame_id))
        prev_pose = find_camera_location(db.get_absolute_extrinsics(start_frame_id))

        if curr_pose is None or prev_pose is None:
            return False
            
        translation = np.linalg.norm(curr_pose - prev_pose)
        return self.min_translation <= translation <= self.max_translation

class CriteriaEvaluationResult:
    """Represents the result of evaluating criteria on a frame"""
    def __init__(self, is_valid: bool, failed_criteria: List[KeyframeCriterion]):
        self.is_valid = is_valid
        self.failed_criteria = failed_criteria

@dataclass
class WindowParameters:
    """Parameters for window selection and keyframe decision"""
    # Window size constraints
    min_bundle_size: int = DEFAULT_MIN_BUNDLE_SIZE
    max_bundle_size: int = DEFAULT_MAX_BUNDLE_SIZE
    
    # List of criteria for keyframe selection
    criteria: List[KeyframeCriterion] = field(default_factory=list)
    
    @classmethod
    def default(cls) -> 'WindowParameters':
        """Create WindowParameters with default criteria"""
        params = cls()
        params.criteria = [
            FeatureCountCriterion(min_features=DEFAULT_MIN_FEATURES, max_features=DEFAULT_MAX_FEATURES),
            TrackOverlapCriterion(min_overlap=DEFAULT_MIN_OVERLAP),
            TranslationCriterion(min_translation=DEFAULT_MIN_TRANSLATION, max_translation=DEFAULT_MAX_TRANSLATION)
        ]
        return params

class WindowSelector:
    def __init__(self, db: TrackingDB, params: Optional[WindowParameters] = None):
        """
        Initialize window selector with tracking database and parameters
        
        Args:
            db: TrackingDB instance containing tracking information
            params: Optional WindowParameters instance for customization
        """
        self.db = db
        self.params = params or WindowParameters.default()
        self.current_index = 0
        self._last_keyframe = 0
        
    def next_window(self) -> Optional[Tuple[int, int]]:
        """
        Find the next window for bundle adjustment. Attempts to maximize window size
        while maintaining valid criteria at the end frame.
        
        Returns:
            Tuple of (start_frame, end_frame) or None if no more windows.
            Window will be between min_bundle_size and max_bundle_size frames,
            attempting to use the largest possible window.
        """
        start = self.current_index
        last_frame = self.db.frame_num() - 1
        
        # Check if we've reached the end
        if start >= last_frame:
            return None
            
        max_possible_end = min(start + self.params.max_bundle_size, last_frame)
        min_possible_end = min(start + self.params.min_bundle_size, last_frame)
        
        # If we can't form minimum bundle size
        if min_possible_end <= start:
            return None
            
        # Search for valid end frame, starting from maximum possible size
        current_end = max_possible_end
        last_valid_frame = None
        
        while current_end >= min_possible_end:
            eval_result = self._evaluate_frame(current_end)
            
            if eval_result.is_valid:
                self.current_index = current_end
                self._last_keyframe = current_end
                return start, current_end
                
            # First failure marks the boundary for next window
            if not last_valid_frame and current_end > min_possible_end:
                last_valid_frame = current_end - 1
                
            current_end -= 1
            
        # Use minimum window size if no valid frame found
        self.current_index = min_possible_end
        return start, min_possible_end
        
    def peek_next_window(self) -> Optional[Tuple[int, int]]:
        """Preview next window without advancing internal state"""
        curr_index = self.current_index
        next_window = self.find_next_window()
        self.current_index = curr_index
        return next_window
        
    def reset(self) -> None:
        """Reset internal state"""
        self.current_index = 0
        self._last_keyframe = 0
        
    @property
    def current_window(self) -> Optional[Tuple[int, int]]:
        """Get current window bounds"""
        if self.current_index == 0:
            return None
        return (self._last_keyframe, self.current_index)
        
    def _evaluate_frame(self, frame_id: int) -> CriteriaEvaluationResult:
        """
        Evaluate all criteria for a given frame
        
        Args:
            frame_id: Frame to evaluate
            
        Returns:
            CriteriaEvaluationResult containing validity and failed criteria
        """
        if frame_id == 0 or self.current_index == frame_id:
            return CriteriaEvaluationResult(False, [])
            
        failed_criteria = []
        for criterion in self.params.criteria:
            if not criterion.evaluate(self.current_index, frame_id, self.db):
                failed_criteria.append(criterion)
                
        return CriteriaEvaluationResult(len(failed_criteria) == 0, failed_criteria)
    
    def add_criterion(self, criterion: KeyframeCriterion) -> None:
        """Add a new criterion for keyframe selection"""
        self.params.criteria.append(criterion)
        
    def remove_criterion(self, criterion_type: type) -> None:
        """Remove all criteria of specified type"""
        self.params.criteria = [c for c in self.params.criteria 
                              if not isinstance(c, criterion_type)]
