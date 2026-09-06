import argparse
from pathlib import Path
import sys
import time
import cv2

ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.omr_engine import OMREngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI Test Bench for Optical Music Recognition (Stage 2: Routed Transcoda / SMT to ABC)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to an image crop file (.png) or directory containing crops"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(ROOT_DIR / "test_bench" / "output" / "omr_results"),
        help="Directory to save generated .abc files"
    )
    parser.add_argument(
        "--cls", "-c",
        type=str,
        default="auto",
        choices=["auto", "staff", "grand_staff", "system"],
        help="Notation class override (default: 'auto' from filename tag)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        help="Inference device: 'cpu' or 'cuda' (default: 'cpu')"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max generation tokens (default: 512)"
    )
    return parser.parse_args()


def detect_class_from_name(path: Path) -> str:
    name = path.stem.lower()
    if "grand_staff" in name or "grandstaff" in name:
        return "grand_staff"
    elif "system" in name:
        return "system"
    else:
        return "staff"


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" OPTICAL MUSIC RECOGNITION (OMR) TEST BENCH (Stage 2: Transcoda / SMT)")
    print("=" * 70)
    print(f" Input       : {input_path}")
    print(f" Output Dir  : {output_dir}")
    print(f" Device      : {args.device}")
    print(f" Class Mode  : {args.cls}")
    print("-" * 70)

    # Collect crop files
    if input_path.is_file():
        crop_files = [input_path]
    elif input_path.is_dir():
        crop_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
        if not crop_files:
            # Check subdirectories
            crop_files = sorted(list(input_path.rglob("*.png")))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if not crop_files:
        print(f"No image crops found at {input_path}")
        return

    print(f"[1/3] Found {len(crop_files)} crop(s) for transcription.")

    omr = OMREngine(device=args.device)

    total_time = 0.0
    success_count = 0

    try:
        for idx, crop_file in enumerate(crop_files, start=1):
            if args.cls == "auto":
                not_class = detect_class_from_name(crop_file)
            else:
                not_class = args.cls

            print(f"\n[2/3] Processing [{idx}/{len(crop_files)}]: {crop_file.name}")
            print(f"      Routed Class : {not_class}")

            img_bgr = cv2.imread(str(crop_file))
            if img_bgr is None:
                print(f"      Error: Failed to read image: {crop_file}")
                continue

            t0 = time.perf_counter()
            result = omr.transcribe_crop(
                crop_bgr=img_bgr,
                notation_class=not_class,
                title=crop_file.stem,
                max_tokens=args.max_tokens
            )
            elapsed = time.perf_counter() - t0
            total_time += elapsed

            print(f"      Model Used   : {result['model_used']}")
            print(f"      Status       : {result['status']} ({elapsed:.2f}s)")

            if result["status"] == "success":
                success_count += 1
                abc_code = result["abc"]
                out_abc_file = output_dir / f"{crop_file.stem}.abc"
                out_abc_file.write_text(abc_code, encoding="utf-8")
                
                # Save raw kern for reference
                out_kern_file = output_dir / f"{crop_file.stem}.kern"
                out_kern_file.write_text(result["raw_kern"], encoding="utf-8")

                print(f"      Saved ABC    : {out_abc_file}")
                print(f"\n--- ABC PREVIEW ({crop_file.name}) ---")
                preview_lines = abc_code.splitlines()[:12]
                for l in preview_lines:
                    print(f"      {l}")
                if len(abc_code.splitlines()) > 12:
                    print("      ...")
                print("-" * 50)
            else:
                print(f"      Error Details: {result.get('error', 'unknown error')}")

    finally:
        # Mandatory GPU Cleanup
        print("\n[3/3] Performing mandatory VRAM / Memory purge...")
        omr.purge_gpu_memory()
        print("      VRAM completely purged. Ready for VLM Phase.")

    print("\n" + "=" * 70)
    print(f" OMR TEST COMPLETED: {success_count}/{len(crop_files)} crops transcribed successfully.")
    print(f" Total time: {total_time:.2f}s (Avg: {total_time / max(1, len(crop_files)):.2f}s per crop)")
    print(f" Outputs saved in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
