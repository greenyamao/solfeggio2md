import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import pymupdf as fitz

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.page_preprocessor import PagePreprocessor
from core.layout_detector import LayoutDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI Test Bench for Music Page Slicing, Gutter Splitting, and Layout Analysis (OLA v2.0)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input file (PDF or image .png/.jpg)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(ROOT_DIR / "test_bench" / "output"),
        help="Directory to save output crops, debug images, and masks"
    )
    parser.add_argument(
        "--page", "-p",
        type=int,
        default=1,
        help="1-indexed page number if processing a PDF (default: 1)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Rendering DPI for PDF pages (default: 200)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for OLA YOLO detector (default: 0.25)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: 'cpu' or 'cuda' (default: 'cpu')"
    )
    return parser.parse_args()


def load_input_pages(input_path: Path, page_num: int, dpi: int):
    """
    Loads page image(s) from either an image file or PDF.
    Returns list of tuples: [(img_bgr, "page_title")]
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"):
        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"Failed to read image file: {input_path}")
        return [(img, input_path.stem)]
    elif suffix == ".pdf":
        with fitz.open(input_path) as doc:
            total_pages = len(doc)
            if page_num < 1 or page_num > total_pages:
                raise ValueError(f"Invalid page number {page_num}. PDF contains {total_pages} pages.")
            page = doc[page_num - 1]
            pix = page.get_pixmap(dpi=dpi)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
            return [(img_bgr, f"{input_path.stem}_P{page_num:04d}")]
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" MUSIC PAGE SLICER TEST BENCH (Stage 1: Normalization + OLA v2.0)")
    print("=" * 70)
    print(f" Input File : {input_path}")
    print(f" Output Dir : {output_dir}")
    print(f" Device     : {args.device}")
    print(f" Confidence : {args.conf}")
    print("-" * 70)

    # 1. Load raw pages
    raw_pages = load_input_pages(input_path, args.page, args.dpi)
    print(f"[1/4] Loaded {len(raw_pages)} source page(s) successfully.")

    preprocessor = PagePreprocessor(target_dpi=args.dpi)
    detector = LayoutDetector(device=args.device, conf_threshold=args.conf)

    for raw_img, base_title in raw_pages:
        # 2. Normalization & Spread Split
        print(f"\n[2/4] Normalizing & checking page layout for: {base_title}")
        norm_pages = preprocessor.process_image(raw_img)
        print(f"      Result: {len(norm_pages)} page(s) after spread/gutter analysis.")

        for p_idx, p_data in enumerate(norm_pages, start=1):
            page_img = p_data["page_img"]
            spread_type = p_data["spread_type"]
            skew_angle = p_data["skew_angle"]
            sub_title = f"{base_title}_{spread_type}" if spread_type != "single" else base_title

            print(f"\n[3/4] Processing page [{p_idx}/{len(norm_pages)}]: {sub_title}")
            print(f"      Spread Type : {spread_type}")
            print(f"      Deskew Angle: {skew_angle:+.2f}°")
            print(f"      Dimensions  : {page_img.shape[1]}x{page_img.shape[0]} px")

            # 3. OLA v2.0 Layout Detection
            detections = detector.detect(page_img)
            
            # Summary statistics
            grand_count = sum(1 for d in detections if d["class"] == "grand_staff")
            single_count = sum(1 for d in detections if d["class"] == "staff")
            system_count = sum(1 for d in detections if d["class"] == "system")

            print(f"\n[4/4] Detections breakdown for {sub_title}:")
            print(f"      - grand_staff (фортепианные акколады / 2 строчки): {grand_count}")
            print(f"      - single_stave (соло / 1 строчка)                 : {single_count}")
            print(f"      - multi-staff system (партитуры / 3+ строчек)     : {system_count}")
            print(f"      - Total music regions                             : {len(detections)}")

            # 4. Export artifacts
            page_out_dir = output_dir / sub_title
            crops_dir = page_out_dir / "crops"
            page_out_dir.mkdir(parents=True, exist_ok=True)
            crops_dir.mkdir(parents=True, exist_ok=True)

            # Debug image with colored boxes
            debug_img = detector.render_debug_image(page_img, detections)
            debug_path = page_out_dir / f"{sub_title}_debug_boxes.png"
            cv2.imwrite(str(debug_path), debug_img)
            print(f"\n      Saved Debug Visual : {debug_path}")

            # Masked page and crops
            masked_img, crops_data = detector.mask_page(page_img, detections, tag_prefix=sub_title)
            masked_path = page_out_dir / f"{sub_title}_masked.png"
            cv2.imwrite(str(masked_path), masked_img)
            print(f"      Saved Masked Page  : {masked_path}")

            for c in crops_data:
                crop_path = crops_dir / f"{c['stub_id']}.png"
                cv2.imwrite(str(crop_path), c["crop_img"])
                print(f"      - Crop: {crop_path.name} ({c['class']}, conf={c['confidence']:.2f})")

    print("\n" + "=" * 70)
    print(" TEST COMPLETED SUCCESSFULLY!")
    print(f" All outputs generated in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
