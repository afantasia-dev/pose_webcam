import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import csv
import time
import math
import threading
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
CSV_PATH = "hand_poses.csv"

ROI_SIZE = 200

# Umbral: mientras más bajo, más estricto. Ajusta según tu cámara/uso.
# 0.15-0.35 suele ser un rango razonable si guardas x,y,z normalizados tal cual.
MATCH_THRESHOLD = 0.22

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

def flatten_hand_landmarks(hand_landmarks) -> List[float]:
    """Convierte 21 landmarks en un vector [x1,y1,z1,x2,y2,z2,...]."""
    v = []
    for lm in hand_landmarks:
        v.extend([float(lm.x), float(lm.y), float(lm.z)])
    return v  # length = 63

def euclidean(a: List[float], b: List[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)

def load_poses(csv_path: str) -> List[List[float]]:
    poses: List[List[float]] = []
    if not os.path.exists(csv_path):
        return poses
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                poses.append([float(x) for x in row])
            except ValueError:
                # ignora filas no numéricas
                continue
    return poses

def append_pose(csv_path: str, pose_vec: List[float]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(pose_vec)

def best_match(pose_vec: List[float], database: List[List[float]]) -> Tuple[Optional[int], float]:
    """Devuelve (indice_mejor, distancia). Si no hay datos => (None, inf)."""
    if not database:
        return None, float("inf")
    best_i = None
    best_d = float("inf")
    for i, ref in enumerate(database):
        if len(ref) != len(pose_vec):
            continue
        d = euclidean(pose_vec, ref)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d

class SharedResult:
    def __init__(self):
        self.lock = threading.Lock()
        self.result: Optional[vision.HandLandmarkerResult] = None

shared = SharedResult()

def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    with shared.lock:
        shared.result = result

def main():
    # Carga poses existentes del CSV
    database = load_poses(CSV_PATH)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,  # para simplificar: grabamos/comparamos solo la mano "principal"
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=result_callback,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

    WINDOW_NAME = "macros landmarker"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # ventana redimensionable
    cv2.resizeWindow(WINDOW_NAME, 1080, 1920)    
    
    if not cap.isOpened():
        raise RuntimeError("No pude abrir la webcam (index 0).")

    last_saved_ts = 0.0  # anti-rebote al guardar

    try:
        
       
              # tamaño inicial (ancho, alto)
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_bgr = cv2.flip(frame_bgr, 1)
            h, w = frame_bgr.shape[:2]

            # ROI centrado
            cx, cy = w // 2, h // 2
            half = ROI_SIZE // 2
            x1, y1 = cx - half, cy - half
            x2, y2 = cx + half, cy + half

            # Enviar frame al landmarker (async)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            landmarker.detect_async(mp_image, int(time.time() * 1000))

            # Traer último resultado
            with shared.lock:
                result = shared.result

            current_pose_vec = None
            hand_inside_roi = False

            # Dibujar mano con OpenCV + extraer vector de pose
            if result and result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]  # primera mano
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                # check ROI: "any landmark inside"
                hand_inside_roi = any(point_in_rect(px, py, x1, y1, x2, y2) for (px, py) in pts)

                # dibujar puntos
                for (px, py) in pts:
                    cv2.circle(frame_bgr, (px, py), 3, (0, 255, 0), -1)
                # dibujar conexiones
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame_bgr, pts[a], pts[b], (0, 200, 255), 2)

                current_pose_vec = flatten_hand_landmarks(hand_landmarks)

            # Matching contra CSV
            match_text = "no hand"
            roi_color = (0, 0, 255)

            if current_pose_vec is not None:
                best_i, best_d = best_match(current_pose_vec, database)
                is_match = best_d <= MATCH_THRESHOLD

                match_text = f"match={is_match}  d={best_d:.3f}  saved={len(database)}"
                roi_color = (0, 255, 0) if (hand_inside_roi and is_match) else ((0, 255, 255) if hand_inside_roi else (0, 0, 255))

            # Dibujar ROI + texto
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), roi_color, 2)
            cv2.putText(frame_bgr, match_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_bgr, "Press 'r' to record, 'ESC' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

            rotated_image = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow("Hands pose record/match", rotated_image)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            # Guardar con 'r'
            if key == ord("r"):
                now = time.time()
                if now - last_saved_ts < 0.5:
                    continue  # anti-rebote

                if current_pose_vec is None:
                    print("No hand detected; nothing to record.")
                else:
                    append_pose(CSV_PATH, current_pose_vec)
                    database.append(current_pose_vec)
                    last_saved_ts = now
                    print(f"Saved pose #{len(database)-1} to {CSV_PATH}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

if __name__ == "__main__":
    main()
