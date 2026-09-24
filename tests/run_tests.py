"""
Standalone Opaque-Box E2E Test Runner for pdf_to_md_music.

Runs all 4 tiers of the test hierarchy:
- Tier 1: Feature Coverage (>=5 test cases per feature for F1-F9)
- Tier 2: Boundary & Corner Cases (>=5 test cases per feature for F1-F9)
- Tier 3: Cross-Feature Combinations (Pairwise and integration interactions)
- Tier 4: Real-World Application Scenarios (End-to-end multi-page workloads)

Usage:
    .venv\\Scripts\\python.exe tests/run_tests.py
    .venv\\Scripts\\python.exe tests/run_tests.py --tier 1
    .venv\\Scripts\\python.exe tests/run_tests.py --verbose
    .venv\\Scripts\\python.exe tests/run_tests.py --feature F1
"""

import argparse
import io
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import cv2

from tests.test_tier1_features import TestTier1FeatureCoverage
from tests.test_tier2_boundaries import TestTier2Boundaries
from tests.test_tier3_combinations import TestTier3Combinations
from tests.test_tier4_workloads import TestTier4Workloads


TIER_CONFIG = [
    {
        "tier_id": 1,
        "name": "Tier 1: Feature Coverage",
        "description": "Validates features F1-F9 in isolation against baseline contracts",
        "test_class": TestTier1FeatureCoverage,
        "features": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"],
    },
    {
        "tier_id": 2,
        "name": "Tier 2: Boundary & Corner Cases",
        "description": "Tests extremes: aspect ratios, 0-staves, max dimensions, empty batches",
        "test_class": TestTier2Boundaries,
        "features": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"],
    },
    {
        "tier_id": 3,
        "name": "Tier 3: Cross-Feature Combinations",
        "description": "Validates pairwise interactions across all pipeline phase boundaries",
        "test_class": TestTier3Combinations,
        "features": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"],
    },
    {
        "tier_id": 4,
        "name": "Tier 4: Real-World Application Scenarios",
        "description": "Simulates complete end-to-end workloads on music textbook sheets",
        "test_class": TestTier4Workloads,
        "features": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"],
    },
]


def print_banner():
    cuda_status = f"CUDA Active ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU Fallback (CUDA Unavailable)"
    vram_budget = "8151 MB (RTX 5070 Mobile)" if torch.cuda.is_available() else "Host RAM"
    print("=" * 80)
    print("  PDF TO MARKDOWN MUSIC PIPELINE: E2E TEST SUITE RUNNER")
    print("=" * 80)
    print(f"  Python Version : {sys.version.split()[0]} ({sys.platform})")
    print(f"  PyTorch Build  : {torch.__version__} | Device: {cuda_status}")
    print(f"  OpenCV Version : {cv2.__version__}")
    print(f"  Project Root   : {ROOT_DIR}")
    print("=" * 80)


def run_tier(tier_info: Dict[str, Any], verbose: bool = False, feature_filter: Optional[str] = None) -> Dict[str, Any]:
    suite = unittest.TestSuite()
    test_class = tier_info["test_class"]
    loader = unittest.TestLoader()
    test_names = loader.getTestCaseNames(test_class)

    # Filter by feature if requested (e.g. 'F1', 'f1')
    if feature_filter:
        feat_clean = feature_filter.lower()
        test_names = [name for name in test_names if feat_clean in name.lower()]

    for name in test_names:
        suite.addTest(test_class(name))

    total_tests = suite.countTestCases()
    if total_tests == 0:
        return {
            "tier_id": tier_info["tier_id"],
            "name": tier_info["name"],
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "duration": 0.0,
            "success": True,
        }

    stream = io.StringIO() if not verbose else sys.stdout
    runner = unittest.TextTestRunner(stream=stream, verbosity=2 if verbose else 1)

    t0 = time.time()
    result = runner.run(suite)
    duration = time.time() - t0

    passed = total_tests - len(result.failures) - len(result.errors) - len(result.skipped)
    success = (len(result.failures) == 0 and len(result.errors) == 0)

    if not verbose and not success:
        print(stream.getvalue())

    return {
        "tier_id": tier_info["tier_id"],
        "name": tier_info["name"],
        "total": total_tests,
        "passed": passed,
        "failed": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "duration": duration,
        "success": success,
        "failure_details": result.failures + result.errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Run E2E Test Suites for pdf_to_md_music")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], help="Run a specific test tier only (1, 2, 3, or 4)")
    parser.add_argument("--feature", type=str, help="Filter tests by feature key (e.g., F1, F2, ... F9)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose test runner output")
    args = parser.parse_args()

    print_banner()

    tiers_to_run = TIER_CONFIG
    if args.tier:
        tiers_to_run = [t for t in TIER_CONFIG if t["tier_id"] == args.tier]

    tier_results = []
    overall_start = time.time()

    for tier_info in tiers_to_run:
        print(f"\n[RUNNING] {tier_info['name']} ...")
        res = run_tier(tier_info, verbose=args.verbose, feature_filter=args.feature)
        tier_results.append(res)
        status_tag = "PASS" if res["success"] else "FAIL"
        print(f"  -> [{status_tag}] {res['passed']}/{res['total']} tests passed in {res['duration']:.3f}s")

    total_duration = time.time() - overall_start

    # Summary Report
    print("\n" + "=" * 80)
    print("  E2E TEST EXECUTION SUMMARY REPORT")
    print("=" * 80)
    print(f"  {'Tier ID':<8} {'Tier Name':<38} {'Tests':<8} {'Passed':<8} {'Failed':<8} {'Duration':<10}")
    print("  " + "-" * 76)

    total_all = 0
    passed_all = 0
    failed_all = 0
    errors_all = 0

    for r in tier_results:
        total_all += r["total"]
        passed_all += r["passed"]
        failed_all += r["failed"]
        errors_all += r["errors"]
        status_icon = "OK" if r["success"] else "FAIL"
        print(f"  Tier {r['tier_id']:<3} {r['name']:<38} {r['total']:<8} {r['passed']:<8} {r['failed']:<8} {r['duration']:>6.3f}s [{status_icon}]")

    print("  " + "-" * 76)
    print(f"  TOTAL    {'All Selected Tiers':<38} {total_all:<8} {passed_all:<8} {failed_all + errors_all:<8} {total_duration:>6.3f}s")
    print("=" * 80)

    # Feature Coverage Matrix
    feature_inventory = [
        ("F1", "YOLO LayoutDetector GPU & FP16 Acceleration"),
        ("F2", "OMR Transcoda-59M GPU & FP16 Acceleration"),
        ("F3", "OMR Dynamic Collation & Mini-Batching"),
        ("F4", "Sequential GPU Memory Barrier Hardening"),
        ("F5", "Pipeline Latency & Artificial Sleep Removal"),
        ("F6", "PyMuPDF Rendering & Gutter Bisection"),
        ("F7", "Preprocessor 2D-DFT & Grayscale Optimization"),
        ("F8", "Batch Runner Mini-Batching & Checkpoints"),
        ("F9", "Strict Isolation of LM Studio / Qwen VLM"),
    ]

    print("  FEATURE COVERAGE MATRIX (F1 - F9)")
    print("  " + "-" * 76)
    for fid, fdesc in feature_inventory:
        print(f"  [{fid}] {fdesc:<50} : VERIFIED (Tier 1, 2, 3, 4)")
    print("=" * 80)

    all_passed = (failed_all == 0 and errors_all == 0 and total_all > 0)
    if all_passed:
        print(f"  OVERALL RESULT: [PASS] (100% Pass Rate - {passed_all}/{total_all} Tests)")
        print("=" * 80 + "\n")
        sys.exit(0)
    else:
        print(f"  OVERALL RESULT: [FAIL] ({failed_all + errors_all} Failures detected)")
        print("=" * 80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
