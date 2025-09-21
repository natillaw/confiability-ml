import cv2
import os

# Rutas
video_dir = r"C:\Users\Natilla\Music\MACHINE LEARNING\data\train"
output_dir = r"C:\Users\Natilla\Music\MACHINE LEARNING\data\train_frames"

os.makedirs(output_dir, exist_ok=True)

# Cantidad de frames por segundo de video a extraer
frames_por_video = 10  # ej. 1 por segundo si los videos duran 10s

for clase in ['lie', 'truth']:
    clase_dir = os.path.join(video_dir, clase)
    output_clase_dir = os.path.join(output_dir, clase)
    os.makedirs(output_clase_dir, exist_ok=True)

    for video in os.listdir(clase_dir):
        if not video.lower().endswith(".mp4"):
            continue
        video_path = os.path.join(clase_dir, video)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        frames_interval = int(fps)  # 1 por segundo
        count = 0
        saved = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frames_interval == 0:
                frame_name = f"{os.path.splitext(video)[0]}_f{count}.jpg"
                save_path = os.path.join(output_clase_dir, frame_name)
                cv2.imwrite(save_path, frame)
                saved += 1
                if saved >= frames_por_video:
                    break
            count += 1
        cap.release()
