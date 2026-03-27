import argparse
import os
import sys
from argparse import Namespace

import torch
from av_hubert.fairseq.fairseq import checkpoint_utils
import av_hubert.fairseq.fairseq.utils as fairseq_utils


def ensure_avhubert_import_paths() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    avhubert_root = os.path.join(repo_root, "av_hubert")
    avhubert_pkg_dir = os.path.join(avhubert_root, "avhubert")

    if avhubert_root not in sys.path:
        sys.path.insert(0, avhubert_root)
    if avhubert_pkg_dir not in sys.path:
        sys.path.insert(0, avhubert_pkg_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-video SpeechForensics inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default=None,
        help="README-style: video root dir",
    )
    parser.add_argument(
        "--mouth_dir",
        type=str,
        default=None,
        help="README-style: cropped mouth dir",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="README-style: relative video path (same as first column in file_list)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/large_vox_iter5.pt",
        help="AV-HuBERT checkpoint path",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Maximum seconds consumed by model",
    )
    return parser.parse_args()


def resolve_input_paths(args: argparse.Namespace) -> tuple[str, str]:
    if not args.video_root or not args.mouth_dir:
        raise ValueError("--video_root and --mouth_dir are required.")

    video_root = os.path.abspath(args.video_root)
    mouth_dir = os.path.abspath(args.mouth_dir)

    if os.path.isabs(args.video_path):
        video_full = os.path.abspath(args.video_path)
    else:
        video_full = os.path.abspath(os.path.join(video_root, args.video_path))

    if os.path.commonpath([video_root, video_full]) != video_root:
        raise ValueError("--video_path must be under --video_root.")

    relative_video_path = os.path.relpath(video_full, video_root)
    mouth_path = os.path.abspath(os.path.join(mouth_dir, relative_video_path))
    wav_path = os.path.splitext(mouth_path)[0] + ".wav"
    return mouth_path, wav_path


def validate_inputs(mouth_path: str, wav_path: str, checkpoint_path: str) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(mouth_path):
        raise FileNotFoundError(f"Mouth video not found: {mouth_path}")
    if not mouth_path.lower().endswith(".mp4"):
        raise ValueError("Resolved mouth path must be an .mp4 file")
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV not found: {wav_path}")
    if not wav_path.lower().endswith(".wav"):
        raise ValueError("Resolved WAV path must be a .wav file")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by the current evaluate.py implementation (uses .cuda(0) internally)."
        )


def load_model(checkpoint_path: str):
    fairseq_utils.import_user_module(Namespace(user_dir=os.getcwd()))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])

    model = models[0]
    if hasattr(models[0], "decoder"):
        print("Checkpoint: fine-tuned")
        model = models[0].encoder.w2v_model
    else:
        print("Checkpoint: pre-trained w/o fine-tuning")

    model = model.cuda().eval()
    return model, task


def main() -> None:
    args = parse_args()
    ensure_avhubert_import_paths()

    import evaluate

    mouth_path, wav_path = resolve_input_paths(args)

    validate_inputs(mouth_path, wav_path, args.checkpoint_path)

    model, task = load_model(args.checkpoint_path)

    # evaluate.py uses module-level globals for model and task.
    evaluate.model = model
    evaluate.task = task

    score = evaluate.evaluate_audio_visual_feature(mouth_path, wav_path, max_length=args.max_length)
    print(f"Mouth video: {mouth_path}")
    print(f"WAV audio : {wav_path}")
    print(f"AV sync score: {score:.6f}")


if __name__ == "__main__":
    main()
