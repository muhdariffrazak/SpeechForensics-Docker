"""
AV-HuBERT Speech Forensics Inference Script

This script evaluates audio-visual synchronization for a single video using
a pre-trained AV-HuBERT model. It requires cropped mouth videos and corresponding
audio files.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from av_hubert.fairseq.fairseq import checkpoint_utils
import av_hubert.fairseq.fairseq.utils as fairseq_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelPaths:
    """Container for model and data paths."""
    checkpoint_path: Path
    mouth_video_path: Path
    audio_path: Path


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    max_seconds: int = 50
    use_gpu: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive, got {self.max_seconds}")


class AVHubertModelLoader:
    """Handles loading and preparation of AV-HuBERT models."""
    
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self._validate_checkpoint()
    
    def _validate_checkpoint(self) -> None:
        """Ensure checkpoint file exists."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )
    
    @staticmethod
    def _setup_user_modules() -> None:
        """Import user modules for custom model architectures."""
        fairseq_utils.import_user_module(
            argparse.Namespace(user_dir=str(Path.cwd()))
        )
    
    def load(self) -> Tuple[torch.nn.Module, object]:
        """
        Load and prepare model for inference.
        
        Returns:
            Tuple of (model, task) ready for evaluation
        """
        self._setup_user_modules()
        
        # Load model ensemble and task
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [str(self.checkpoint_path)]
        )
        
        # Extract the actual model (handle both pre-trained and fine-tuned)
        model = self._extract_core_model(models[0])
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model moved to GPU")
        else:
            logger.warning("CUDA not available - using CPU (may be slow)")
        
        model.eval()
        return model, task
    
    @staticmethod
    def _extract_core_model(model: torch.nn.Module) -> torch.nn.Module:
        """
        Extract the core AV-HuBERT model from potential wrapper.
        
        Fine-tuned models have a decoder wrapper; pre-trained models don't.
        """
        if hasattr(model, "decoder"):
            logger.info("Loaded fine-tuned model (extracting encoder)")
            return model.encoder.w2v_model
        else:
            logger.info("Loaded pre-trained model")
            return model


class InputResolver:
    """Handles input path resolution and validation."""
    
    def __init__(self, video_root: Path, mouth_dir: Path):
        """
        Initialize resolver with base directories.
        
        Args:
            video_root: Root directory containing original videos
            mouth_dir: Root directory containing cropped mouth videos
        """
        self.video_root = video_root.resolve()
        self.mouth_dir = mouth_dir.resolve()
        
        self._validate_directories()
    
    def _validate_directories(self) -> None:
        """Ensure base directories exist."""
        if not self.video_root.exists():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")
        if not self.mouth_dir.exists():
            raise FileNotFoundError(f"Mouth directory not found: {self.mouth_dir}")
    
    def resolve_paths(self, video_path: str) -> ModelPaths:
        """
        Resolve full paths for mouth video and audio.
        
        Args:
            video_path: Path to video (relative to video_root or absolute)
            
        Returns:
            ModelPaths object with resolved file paths
        """
        video_full = self._resolve_video_full_path(video_path)
        self._validate_video_location(video_full)
        
        relative_path = video_full.relative_to(self.video_root)
        
        mouth_path = self.mouth_dir / relative_path
        audio_path = mouth_path.with_suffix(".wav")
        
        return ModelPaths(
            checkpoint_path=Path(),  # Will be set later
            mouth_video_path=mouth_path,
            audio_path=audio_path
        )
    
    def _resolve_video_full_path(self, video_path: str) -> Path:
        """Resolve video path to absolute path."""
        path = Path(video_path)
        if path.is_absolute():
            return path.resolve()
        return (self.video_root / path).resolve()
    
    def _validate_video_location(self, video_full: Path) -> None:
        """
        Ensure video path is under video_root (security check).
        
        Raises:
            ValueError: If video is outside video_root
        """
        try:
            video_full.relative_to(self.video_root)
        except ValueError:
            raise ValueError(
                f"Video path {video_full} is not under video root {self.video_root}"
            )
    
    @staticmethod
    def validate_input_files(paths: ModelPaths) -> None:
        """Validate that all required input files exist."""
        if not paths.mouth_video_path.exists():
            raise FileNotFoundError(
                f"Mouth video not found: {paths.mouth_video_path}"
            )
        
        if not paths.audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {paths.audio_path}"
            )
        
        # Validate file extensions
        if paths.mouth_video_path.suffix.lower() != ".mp4":
            raise ValueError(
                f"Mouth video must be MP4 format: {paths.mouth_video_path}"
            )
        
        if paths.audio_path.suffix.lower() != ".wav":
            raise ValueError(
                f"Audio must be WAV format: {paths.audio_path}"
            )


class InferenceEngine:
    """Handles the actual inference process."""
    
    def __init__(self, evaluate_module, config: InferenceConfig):
        """
        Initialize inference engine.
        
        Args:
            evaluate_module: The evaluate module from av_hubert
            config: Inference configuration
        """
        self.evaluate = evaluate_module
        self.config = config
    
    def compute_sync_score(
        self, 
        paths: ModelPaths, 
        model: torch.nn.Module, 
        task: object
    ) -> float:
        """
        Compute audio-visual synchronization score.
        
        Args:
            paths: ModelPaths containing mouth video and audio paths
            model: Loaded AV-HuBERT model
            task: Fairseq task object
            
        Returns:
            Float score (higher = better synchronization)
        """
        # Configure evaluate module (uses module-level globals)
        self.evaluate.model = model
        self.evaluate.task = task
        
        logger.info("Computing AV sync score...")
        score = self.evaluate.evaluate_audio_visual_feature(
            str(paths.mouth_video_path),
            str(paths.audio_path),
            max_length=self.config.max_seconds
        )
        
        return score


class ArgumentParser:
    """Handles command-line argument parsing."""
    
    @staticmethod
    def parse() -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Run single-video SpeechForensics inference",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        
        # Input directories
        parser.add_argument(
            "--video_root",
            type=str,
            required=True,
            help="Root directory containing original videos",
        )
        parser.add_argument(
            "--mouth_dir",
            type=str,
            required=True,
            help="Root directory containing cropped mouth videos",
        )
        
        # Video specification
        parser.add_argument(
            "--video_path",
            type=str,
            required=True,
            help="Path to video (relative to video_root or absolute)",
        )
        
        # Model configuration
        parser.add_argument(
            "--checkpoint_path",
            type=str,
            default="checkpoints/large_vox_iter5.pt",
            help="AV-HuBERT checkpoint path",
        )
        parser.add_argument(
            "--max_seconds",
            type=int,
            default=50,
            help="Maximum video duration in seconds to process",
        )
        
        return parser.parse_args()


def setup_import_paths() -> None:
    """Add AV-HuBERT modules to Python path."""
    repo_root = Path(__file__).parent.resolve()
    avhubert_root = repo_root / "av_hubert"
    avhubert_pkg_dir = avhubert_root / "avhubert"
    
    # Add to path if not already present
    for path in [avhubert_root, avhubert_pkg_dir]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            logger.debug(f"Added to sys.path: {path_str}")


def main() -> None:
    """Main execution function."""
    try:
        # Parse arguments
        args = ArgumentParser.parse()
        
        # Setup environment
        setup_import_paths()
        
        # Import evaluate module (must happen after path setup)
        import evaluate
        
        # Create configuration
        config = InferenceConfig(max_seconds=args.max_seconds)
        config.validate()
        
        # Setup input resolver
        resolver = InputResolver(
            video_root=Path(args.video_root),
            mouth_dir=Path(args.mouth_dir)
        )
        
        # Resolve input paths
        paths = resolver.resolve_paths(args.video_path)
        paths.checkpoint_path = Path(args.checkpoint_path)
        
        # Validate inputs
        resolver.validate_input_files(paths)
        
        # Load model
        logger.info(f"Loading model from {paths.checkpoint_path}")
        loader = AVHubertModelLoader(paths.checkpoint_path)
        model, task = loader.load()
        
        # Run inference
        engine = InferenceEngine(evaluate, config)
        score = engine.compute_sync_score(paths, model, task)
        
        # Output results
        logger.info("=" * 50)
        logger.info("Inference Results:")
        logger.info(f"Mouth video: {paths.mouth_video_path}")
        logger.info(f"Audio file: {paths.audio_path}")
        logger.info(f"AV sync score: {score:.6f}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()