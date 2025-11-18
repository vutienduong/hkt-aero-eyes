"""Check if all annotation frame numbers are valid for their videos."""
import json
import cv2
from pathlib import Path

# Load annotations
ann_path = Path("data/train/annotations/annotations.json")
with open(ann_path) as f:
    annotations = json.load(f)

print("Checking video frame counts vs annotations...\n")

issues = []
for item in annotations:
    video_id = item["video_id"]
    video_path = Path(f"data/train/samples/{video_id}/drone_video.mp4")

    if not video_path.exists():
        print(f"❌ {video_id}: Video file not found")
        continue

    # Get frame count
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Check all frame numbers in annotations
    max_frame = 0
    for interval in item["annotations"]:
        for bbox in interval["bboxes"]:
            frame_num = bbox["frame"]
            max_frame = max(max_frame, frame_num)

            if frame_num >= total_frames:
                issues.append({
                    "video_id": video_id,
                    "frame": frame_num,
                    "total_frames": total_frames
                })

    status = "✓" if max_frame < total_frames else "❌"
    print(f"{status} {video_id}: max_frame={max_frame}, total_frames={total_frames}")

if issues:
    print(f"\n⚠️  Found {len(issues)} problematic frames:")
    for issue in issues[:10]:  # Show first 10
        print(f"  {issue['video_id']}: frame {issue['frame']} >= {issue['total_frames']}")
else:
    print("\n✓ All annotations are within video bounds!")
