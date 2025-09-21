# evaluar_video_haar_preciso.py
import os
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# =========================
# CONFIG (igual que antes)
# =========================
MODEL_PATH     = r"C:\Users\Natilla\Music\MACHINE LEARNING\modelo_confiabilidad.pth"
IMAGE_SIZE     = 64
MAX_FRAMES_AVG = 600

# Haar (detector robusto sin frontalidad "dura")
HAAR_SCALE_FACTOR  = 1.06
HAAR_MIN_NEIGHBORS = 8
MIN_FACE_FRAC      = 0.12
ASPECT_MIN, ASPECT_MAX = 0.75, 1.35

# Visual
BOX_COLOR_POS = (0, 200, 0)
BOX_COLOR_NEG = (0, 0, 255)

# =========================
# MODELO (igual que entrenaste)
# =========================
class RedConfiabilidad(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = RedConfiabilidad().to(device)
state  = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# =========================
# Apertura robusta de video
# =========================
def open_video(path_like):
    p = os.fspath(os.path.normpath(path_like))
    cap = cv2.VideoCapture(p)
    if cap.isOpened(): return cap
    cap.release()
    cap = cv2.VideoCapture(p, cv2.CAP_FFMPEG)
    if cap.isOpened(): return cap
    cap.release()
    cap = cv2.VideoCapture(p, cv2.CAP_ANY)
    if cap.isOpened(): return cap
    cap.release()
    raise FileNotFoundError(f"No se pudo abrir el video: {p}")

# =========================
# CARGA HAAR
# =========================
def _cv2_haar(name: str) -> str:
    local = rf"C:\Users\Natilla\Music\MACHINE LEARNING\.venv\lib\site-packages\cv2\data\{name}"
    if os.path.exists(local):
        return local
    from cv2 import data as cv2data
    return os.path.join(cv2data.haarcascades, name)

face_cascade = cv2.CascadeClassifier(_cv2_haar("haarcascade_frontalface_default.xml"))
if face_cascade.empty():
    raise FileNotFoundError("No se pudo cargar haarcascade_frontalface_default.xml")

# =========================
# UTILS DETECCIÓN/CLASIFICACIÓN
# =========================
def _enhance_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    return gray

def _nms_xyxy(boxes, scores, iou_thr=0.35):
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1,y1,x2,y2 = boxes.T
    areas = (x2-x1+1)*(y2-y1+1)
    idxs = scores.argsort()[::-1]
    keep=[]
    while len(idxs)>0:
        i=idxs[0]; keep.append(int(i))
        xx1=np.maximum(x1[i],x1[idxs[1:]]); yy1=np.maximum(y1[i],y1[idxs[1:]])
        xx2=np.minimum(x2[i],x2[idxs[1:]]); yy2=np.minimum(y2[i],y2[idxs[1:]])
        w=np.maximum(0,xx2-xx1+1); h=np.maximum(0,yy2-yy1+1)
        inter=w*h
        iou=inter/(areas[i]+areas[idxs[1:]]-inter+1e-6)
        idxs=idxs[1:][iou<iou_thr]
    return [tuple(map(int, boxes[k])) for k in keep]

def _classify_face(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1)[0,1].item()
    return float(prob*100.0)

def _detect_primary_face(frame_bgr):
    h, w = frame_bgr.shape[:2]
    gray = _enhance_gray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    min_side = int(max(48, min(h, w)*MIN_FACE_FRAC))

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=HAAR_SCALE_FACTOR,
        minNeighbors=HAAR_MIN_NEIGHBORS,
        minSize=(min_side, min_side)
    )

    boxes=[]; scores=[]
    for (x,y,fw,fh) in faces:
        ar = fw/max(1.0,fh)
        if not (ASPECT_MIN<=ar<=ASPECT_MAX): continue
        boxes.append((x,y,x+fw,y+fh))
        scores.append(fw*fh)

    boxes = _nms_xyxy(boxes, scores, iou_thr=0.35)
    if not boxes: return None
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    return boxes[int(np.argmax(areas))]

# =========================
# OPCIONAL: MediaPipe Face Mesh para yaw/pitch
# =========================
MP_AVAILABLE = False
try:
    import mediapipe as mp
    MP_AVAILABLE = True
    _mp_mesh = mp.solutions.face_mesh
except Exception:
    MP_AVAILABLE = False
    _mp_mesh = None

# Puntos 3D de un modelo facial simplificado (mm) para solvePnP
# (nariz, ojos, comisuras, mentón)
_MODEL_3D = np.array([
    [0.0,   0.0,   0.0],    # nariz (aprox)
    [-30.0, -5.0, -20.0],   # ojo dcho ext
    [ 30.0, -5.0, -20.0],   # ojo izq ext
    [-20.0, 30.0, -20.0],   # comisura dcha
    [ 20.0, 30.0, -20.0],   # comisura izq
    [ 0.0,  65.0,  0.0],    # mentón
], dtype=np.float32)

# Índices de Face Mesh (MediaPipe) aproximados:
#  1: nariz (tip)  |  33: ojo derecho externo | 263: ojo izquierdo externo
#  61: boca derecha | 291: boca izquierda | 199: mentón
_MESH_IDX = [1, 33, 263, 61, 291, 199]

def _estimate_yaw_pitch_from_mesh(frame_bgr, box_xyxy=None):
    """
    Devuelve (yaw_deg, pitch_deg) o (None, None) si no disponible/falla.
    Yaw+: mira hacia su izquierda; Pitch+: mira hacia arriba.
    """
    if not MP_AVAILABLE:
        return None, None

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Para estabilidad, analizar toda la imagen (MediaPipe usa su propio detector)
    with _mp_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as mesh:

        res = mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None, None

        lm = res.multi_face_landmarks[0].landmark
        pts_2d = []
        try:
            for idx in _MESH_IDX:
                p = lm[idx]
                pts_2d.append([p.x * w, p.y * h])
        except Exception:
            return None, None
        pts_2d = np.array(pts_2d, dtype=np.float32)

        # Cámara pinhole aproximada
        focal = w
        cam_mat = np.array([[focal, 0, w/2],
                            [0, focal, h/2],
                            [0, 0, 1]], dtype=np.float32)
        dist = np.zeros((4,1), dtype=np.float32)

        ok, rvec, tvec = cv2.solvePnP(_MODEL_3D, pts_2d, cam_mat, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None

        R, _ = cv2.Rodrigues(rvec)
        # Extracción de ángulos (yaw, pitch, roll) desde R
        sy = np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
        singular = sy < 1e-6
        if not singular:
            pitch = np.degrees(np.arctan2(-R[2,0], sy))
            yaw   = np.degrees(np.arctan2(R[1,0], R[0,0]))
            roll  = np.degrees(np.arctan2(R[2,1], R[2,2]))
        else:
            pitch = np.degrees(np.arctan2(-R[2,0], sy))
            yaw   = np.degrees(np.arctan2(-R[0,1], R[1,1]))
            roll  = 0.0

        return float(yaw), float(pitch)

def frontal_score_from_yaw_pitch(yaw, pitch):
    """F en [0,1]; 1=frontal, 0=desviado. Ajusta divisores para tu caso."""
    if yaw is None or pitch is None:
        return None
    F = 1.0 - abs(yaw)/40.0 - abs(pitch)/30.0
    return float(max(0.0, min(1.0, F)))

# =========================
# VISUALIZAR (igual: muestra score del modelo)
# =========================
def visualizar_video(video_path: str, window_name="Confiabilidad (Haar, preciso)"):
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_period = 1.0/fps; next_t = time.perf_counter()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok: break

        box = _detect_primary_face(frame)
        if box is not None:
            x1,y1,x2,y2 = box
            crop = frame[y1:y2, x1:x2]
            score = _classify_face(crop) if crop.size>0 else 0.0
            color = BOX_COLOR_POS if score>=50 else BOX_COLOR_NEG
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            cv2.putText(frame, f"{score:.1f}%", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow(window_name, frame)

        next_t += frame_period
        remaining = next_t - time.perf_counter()
        if remaining > 0: key = cv2.waitKey(max(1, int(remaining*1000))) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF
            while remaining < -frame_period and cap.grab():
                next_t += frame_period
                remaining = next_t - time.perf_counter()
        if key in (27, ord('q')): break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break

    cap.release()
    try: cv2.destroyWindow(window_name)
    except cv2.error: pass

# =========================
# EVALUAR (continua; interfaz hace muestreo 2fps)
# =========================
def evaluar_video(video_path: str):
    cap = open_video(video_path)
    confidencias=[]; frame_idx=0
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= MAX_FRAMES_AVG: break
        box = _detect_primary_face(frame)
        if box is not None:
            x1,y1,x2,y2 = box
            crop = frame[y1:y2, x1:x2]
            if crop.size>0:
                confidencias.append(_classify_face(crop))
        frame_idx += 1
    cap.release()
    avg = float(np.median(confidencias)) if confidencias else 0.0
    return avg, min(frame_idx, MAX_FRAMES_AVG)

# ======== nombres que usa la interfaz ========
_detect_primary_face = _detect_primary_face
_classify_face = _classify_face
MP_AVAILABLE_FLAG = MP_AVAILABLE
compute_frontal_score = _estimate_yaw_pitch_from_mesh  # devuelve (yaw, pitch)
frontal_score_from_yaw_pitch = frontal_score_from_yaw_pitch
open_video_backend = open_video


if __name__ == "__main__":
    import os
    import cv2
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from tqdm import tqdm

    BASE_DIR = r"C:\Users\Natilla\Music\MACHINE LEARNING\data\test_frames"
    CLASES = {'lie': 0, 'truth': 1}

    y_true = []
    y_pred = []

    for clase_nombre, clase_valor in CLASES.items():
        carpeta = os.path.join(BASE_DIR, clase_nombre)
        for archivo in tqdm(os.listdir(carpeta), desc=f"Evaluando {clase_nombre}"):
            ruta_img = os.path.join(carpeta, archivo)
            try:
                img = cv2.imread(ruta_img)
                if img is None: continue
                score = _classify_face(img)
                pred = int(score >= 50.0)
                y_true.append(clase_valor)
                y_pred.append(pred)
            except Exception as e:
                print(f"Error con {ruta_img}: {e}")

    if y_pred:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\n=== Métricas de evaluación ===")
        print(f"Accuracy : {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall   : {recall:.3f}")
        print(f"F1 Score : {f1:.3f}")
    else:
        print("No se pudo evaluar: no se generaron predicciones válidas.")


