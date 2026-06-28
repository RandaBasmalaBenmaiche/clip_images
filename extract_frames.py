"""
extract_frames.py

Extract evenly time-spaced frames from a short video and save them as
.jpg images into a class folder under data/, so they can be picked up
by the existing create_embeddings.py pipeline without any changes to it.

Why time-spaced sampling instead of every frame:
A few seconds of video can contain 100+ frames at typical frame rates.
Consecutive frames look almost identical (same angle, milliseconds apart),
so extracting all of them mostly adds near-duplicates rather than new
information. Sampling one frame every N seconds instead gives genuinely
different angles/views as the camera (or object) moves, which is closer
to what the model actually benefits from.

Usage:
    python extract_frames.py <video_path> <class_name> [--interval 0.5] [--max-frames 15]

Examples:
    python extract_frames.py my_video.mp4 class9
    python extract_frames.py my_video.mp4 class9 --interval 0.3 --max-frames 20

Arguments:
    video_path    Path to the video file (.mp4, .mov, .avi, etc.)
    class_name    Name of the class folder under data/ to save frames into.
                  Created automatically if it doesn't exist yet.
    --interval    Seconds between extracted frames (default: 0.5).
                  Smaller = more frames = more variety, but more near-duplicates.
    --max-frames  Safety cap on total frames extracted (default: 20), so a long
                  video doesn't accidentally flood a class with hundreds of images.
"""

import argparse
import os
import sys

import cv2


def extract_frames(video_path, class_name, interval_seconds=0.5, max_frames=20, data_dir="data"):
    if not os.path.isfile(video_path):
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    class_dir = os.path.join(data_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video file: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print("Warning: could not read FPS from video, assuming 30 fps.")
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps if fps else 0

    frame_interval = max(1, int(round(fps * interval_seconds)))

    # Figure out a base name for saved files, e.g. "myvideo_frame_001.jpg"
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    video_basename = "".join(c if c.isalnum() or c in "_-" else "_" for c in video_basename)

    saved_count = 0
    frame_idx = 0

    print(f"Video: {video_path}")
    print(f"Duration: ~{duration_seconds:.1f}s, FPS: {fps:.1f}, total frames: {total_frames}")
    print(f"Sampling every {interval_seconds}s (~every {frame_interval} frames), max {max_frames} frames")
    print(f"Saving into: {class_dir}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        if frame_idx % frame_interval == 0:
            if saved_count >= max_frames:
                print(f"Reached max-frames limit ({max_frames}), stopping early.")
                break

            saved_count += 1
            out_name = f"{video_basename}_frame_{saved_count:03d}.jpg"
            out_path = os.path.join(class_dir, out_name)

            # cv2 reads frames in BGR; cv2.imwrite expects BGR too, so no color conversion needed here.
            success = cv2.imwrite(out_path, frame)
            if success:
                print(f"  saved {out_name}")
            else:
                print(f"  failed to save frame {saved_count}")

        frame_idx += 1

    cap.release()

    print(f"\nDone. Saved {saved_count} frames to {class_dir}")
    if saved_count == 0:
        print("No frames were saved -- check that the video file is valid and not empty.")
    else:
        print("Next steps:")
        print("  1. Open the folder and delete any blurry/bad frames if needed.")
        print("  2. Run: python create_embeddings.py")
        print("  3. Run: python benchmark.py   (to check the new class doesn't get confused with others)")


def main():
    parser = argparse.ArgumentParser(description="Extract time-spaced frames from a video into a data/ class folder.")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("class_name", help="Class folder name under data/ (created if missing)")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between extracted frames (default: 0.5)")
    parser.add_argument("--max-frames", type=int, default=20, help="Maximum number of frames to extract (default: 20)")
    parser.add_argument("--data-dir", default="data", help="Root data directory (default: data)")

    args = parser.parse_args()

    extract_frames(
        video_path=args.video_path,
        class_name=args.class_name,
        interval_seconds=args.interval,
        max_frames=args.max_frames,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
