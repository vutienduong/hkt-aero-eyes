import cv2

def iter_video_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()
