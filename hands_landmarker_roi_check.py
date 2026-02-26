import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import threading
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
ROI_SIZE = 200

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

def point_in_rect(px, py, x1, y1, x2, y2):
    return (x1 <= px <= x2) and (y1 <= py <= y2)

class SharedResult:
    def __init__(self):
        self.lock = threading.Lock()
        self.result = None
        self.timestamp_ms = -1

shared = SharedResult()

def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # Store only the most recent result
    with shared.lock:
        shared.result = result
        shared.timestamp_ms = timestamp_ms




def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=result_callback,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    WINDOW_NAME = "Hand Landmarks (Tasks API)"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # ventana redimensionable
    cv2.resizeWindow(WINDOW_NAME, 1080, 1920)          # tamaño inicial (ancho, alto)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Try another index.")
    try:
        
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_bgr = cv2.flip(frame_bgr, 1)
            h, w = frame_bgr.shape[:2]

            # Centered ROI square
            cx, cy = w // 2, h // 2
            half = ROI_SIZE // 2
            x1, y1 = cx - half, cy - half
            x2, y2 = cx + half, cy + half

            # Send frame to MediaPipe asynchronously
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            # Fetch latest result (if any)
            with shared.lock:
                result = shared.result

            hand_inside = False

            if result and result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                    # "Inside" logic: any landmark inside ROI
                    if any(point_in_rect(px, py, x1, y1, x2, y2) for (px, py) in pts):
                        hand_inside = True

                    # Draw landmarks (OpenCV)
                    for (px, py) in pts:
                        cv2.circle(frame_bgr, (px, py), 3, (0, 255, 0), -1)
                    for a, b in HAND_CONNECTIONS:
                        cv2.line(frame_bgr, pts[a], pts[b], (0, 200, 255), 2)

            roi_color = (0, 255, 0) if hand_inside else (0, 0, 255)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), roi_color, 2)
            cv2.putText(
                frame_bgr,
                "HAND INSIDE" if hand_inside else "hand outside",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                roi_color,
                2,
                cv2.LINE_AA,
            )
            rotated_image = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

            cv2.imshow(WINDOW_NAME, rotated_image)
            if cv2.waitKey(1) & 0xFF == 27:
                if key == ord("q") or key == 27:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

if __name__ == "__main__":

    main()
