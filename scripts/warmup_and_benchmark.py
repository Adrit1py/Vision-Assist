"""
VisionAssist - Heavy Model Warmup & Latency Benchmark Tool
==========================================================
Fulfills Adrit's Action Guide (SYSTEM_ARCHITECTURE.md Section 9 Step 1):
Warms up PyTorch, Ultralytics YOLOv8, and EasyOCR on Adrit's edge compute laptop.
Measures and logs:
  1. YOLOv8n object detection latency & FPS
  2. EasyOCR text recognition latency
  3. Geometric monocular depth estimation latency
  4. TTS speech queue dispatch latency
"""

import logging
import sys
import time
from pathlib import Path
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from perception.object_detection import ObjectDetector
from perception.ocr import OCRReader
from perception.depth import DepthEstimator
from audio.tts import TextToSpeechEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("VisionAssist.Benchmark")


def run_benchmark():
    print("=" * 70)
    print(" VISIONASSIST — EDGE COMPUTE BENCHMARK & WARMUP (ADRIT'S NODE) ")
    print("=" * 70)

    # 1. Benchmark YOLOv8 Object Detection
    print("\n[1/4] Warming up & Benchmarking YOLOv8 Object Detection...")
    detector = ObjectDetector()
    warmup_ms = detector.warmup()
    print(f"      -> Initial Warmup Latency: {warmup_ms:.1f} ms")

    bench_yolo = detector.benchmark(num_frames=15)
    print(f"      -> Model Mode: {'LIVE YOLOv8' if bench_yolo['is_live_model'] else 'HEURISTIC MOCK'}")
    print(f"      -> Average Latency: {bench_yolo['avg_latency_ms']} ms")
    print(f"      -> Throughput: {bench_yolo['fps']} FPS")

    # 2. Benchmark EasyOCR Reader
    print("\n[2/4] Warming up & Benchmarking EasyOCR Reader...")
    ocr = OCRReader()
    ocr_warmup_ms = ocr.warmup()
    print(f"      -> Initial Warmup Latency: {ocr_warmup_ms:.1f} ms")

    bench_ocr = ocr.benchmark(num_iterations=3)
    print(f"      -> Model Mode: {'LIVE EasyOCR' if bench_ocr['is_live_model'] else 'HEURISTIC MOCK'}")
    print(f"      -> Average OCR Latency: {bench_ocr['avg_latency_ms']} ms")

    # 3. Benchmark Depth & Spatial Estimation
    print("\n[3/4] Benchmarking Monocular Depth & Spatial Math...")
    depth = DepthEstimator()
    dummy_bbox = [180, 120, 340, 420]
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        dist, prox = depth.estimate("chair", dummy_bbox, 480, 640)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    avg_depth_ms = sum(latencies) / len(latencies)
    print(f"      -> Average Depth Math Latency: {avg_depth_ms:.3f} ms (Distance: {dist}m, Proximity: {prox})")

    # 4. Benchmark TTS Audio Dispatch
    print("\n[4/4] Testing Text-To-Speech (TTS) Engine...")
    tts = TextToSpeechEngine(mute=True)  # Mute during automated benchmark
    t0 = time.perf_counter()
    tts.speak("VisionAssist compute node calibrated.", interrupt=False)
    dispatch_latency_ms = (time.perf_counter() - t0) * 1000.0
    print(f"      -> TTS Async Enqueue Latency: {dispatch_latency_ms:.3f} ms")
    tts.stop()

    print("\n" + "=" * 70)
    print(" BENCHMARK SUMMARY FOR JUDGING REPORT ")
    print("=" * 70)
    print(f"  • YOLOv8 Inference:  {bench_yolo['avg_latency_ms']} ms / frame ({bench_yolo['fps']} FPS)")
    print(f"  • EasyOCR Latency:   {bench_ocr['avg_latency_ms']} ms")
    print(f"  • Depth Estimation:  {avg_depth_ms:.3f} ms")
    print(f"  • TTS Dispatch:      {dispatch_latency_ms:.3f} ms")
    print("=" * 70)
    print(" Status: Edge Compute Node Warm & Ready for Live Demo!\n")


if __name__ == "__main__":
    run_benchmark()
