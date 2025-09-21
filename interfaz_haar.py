# interfaz_haar.py
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image as PILImage, ImageTk
import csv
import cv2

# =========================
# CONFIG: muestreo a 2 fps (cada 0.5 s)
# =========================
STEP_SECONDS = 0.5          # 2 frames/seg
EMA_S_ML = 0.6              # suavizado S_ml
EMA_F    = 0.6              # suavizado F

# -------------------------------------------------
# Rutas seguras
# -------------------------------------------------
def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# -------------------------------------------------
# Backend (Haar + opcional FaceMesh)
# -------------------------------------------------
BACKEND_ERROR = ""
try:
    from evaluar_video_haar_preciso import (
        visualizar_video,
        _detect_primary_face as detect_face,
        _classify_face as classify_face,
        open_video_backend as backend_open_video,
        MP_AVAILABLE_FLAG,
        compute_frontal_score,           # devuelve (yaw, pitch) o (None, None)
        frontal_score_from_yaw_pitch,    # yaw,pitch -> F in [0,1]
    )
except Exception as e:
    BACKEND_ERROR = str(e)
    visualizar_video = None
    detect_face = None
    classify_face = None
    backend_open_video = None
    MP_AVAILABLE_FLAG = False
    compute_frontal_score = None
    frontal_score_from_yaw_pitch = None

def open_capture(path_like):
    if backend_open_video:
        return backend_open_video(path_like)
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
    raise FileNotFoundError(f"No se pudo abrir el video:\n{p}")

# =========================
# App Tkinter
# =========================
class HaarApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analizador de Confiabilidad – Haar (Precisión + Mirada opcional)")
        self.geometry("780x620")
        self.resizable(False, False)

        self.video_path = tk.StringVar()
        self.status_var = tk.StringVar(value="Listo.")
        self.result_var = tk.StringVar(value="Confiabilidad: -")
        self.frames_var = tk.StringVar(value="Frames: -")
        self.progress_var = tk.StringVar(value="")
        self.mp_state_var = tk.StringVar(value=("MediaPipe disponible" if MP_AVAILABLE_FLAG else "MediaPipe NO disponible"))

        self.use_gaze_var = tk.BooleanVar(value=False)
        self.alpha_var = tk.DoubleVar(value=0.80)

        self.ok_img_path = resource_path("confiable.png")
        self.bad_img_path = resource_path("poco_confiable.png")
        self._img_tk = None

        self._eval_thread: threading.Thread | None = None
        self._cancel_eval = threading.Event()

        self._build_ui()

        if BACKEND_ERROR:
            self._set_buttons_state(disabled=True)
            messagebox.showerror("Error de backend", BACKEND_ERROR)

        if not MP_AVAILABLE_FLAG:
            self.use_gaze_var.set(False)

    def _build_ui(self):
        top = tk.Frame(self); top.pack(fill="x", pady=8)
        tk.Entry(top, textvariable=self.video_path, width=86).grid(row=0, column=0, padx=8)
        tk.Button(top, text="Cargar video", command=self.cargar_video).grid(row=0, column=1, padx=6)

        actions = tk.Frame(self); actions.pack(pady=6)
        self.btn_evaluar = tk.Button(actions, text="Evaluar (preciso)", width=18, command=self.evaluar_async)
        self.btn_cancelar = tk.Button(actions, text="Cancelar", width=12, command=self.cancelar, state="disabled")
        self.btn_visualizar = tk.Button(actions, text="Visualizar (preciso)", width=18, command=self.visualizar_async)
        self.btn_csv = tk.Button(actions, text="Exportar CSV…", width=16, command=self.exportar_csv_async)
        self.btn_evaluar.grid(row=0, column=0, padx=6)
        self.btn_cancelar.grid(row=0, column=1, padx=6)
        self.btn_visualizar.grid(row=0, column=2, padx=6)
        self.btn_csv.grid(row=0, column=3, padx=6)

        # Opciones de penalización por mirada
        opts = tk.LabelFrame(self, text="Señal auxiliar de mirada (opcional)")
        opts.pack(fill="x", padx=10, pady=8)
        gaze_chk = tk.Checkbutton(opts, text="Penalizar cuando no mira a cámara (MediaPipe Face Mesh)", variable=self.use_gaze_var)
        gaze_chk.grid(row=0, column=0, sticky="w", padx=10, pady=4)
        if not MP_AVAILABLE_FLAG:
            gaze_chk.config(state="disabled")
        tk.Label(opts, text="α (piso de penalización):").grid(row=1, column=0, sticky="w", padx=10)
        alpha_scale = ttk.Scale(opts, from_=0.60, to=0.99, orient="horizontal", variable=self.alpha_var, length=260)
        alpha_scale.grid(row=1, column=0, sticky="e", padx=10)
        tk.Label(opts, textvariable=self.mp_state_var, fg="#666").grid(row=2, column=0, sticky="w", padx=10, pady=(4,6))

        mid = tk.Frame(self); mid.pack(pady=10)
        tk.Label(mid, textvariable=self.result_var, font=("Arial", 16, "bold")).pack()
        tk.Label(mid, textvariable=self.frames_var, fg="#666").pack(pady=(2, 8))
        self.img_label = tk.Label(mid); self.img_label.pack()

        tk.Label(self, textvariable=self.status_var, fg="#555").pack(pady=(12, 2))
        tk.Label(self, textvariable=self.progress_var, fg="#777").pack(pady=(0, 8))

        tip = (f"Evaluación muestreando 2 fps (1 frame cada {STEP_SECONDS:.1f} s). "
               "La señal de ‘mirada’ es auxiliar y opcional. Visualización: cierra con Q / Esc.")
        tk.Label(self, text=tip, fg="#444").pack()

    # ---------- Helpers ----------
    def _set_buttons_state(self, disabled: bool):
        state = "disabled" if disabled else "normal"
        self.btn_evaluar.config(state=state)
        self.btn_visualizar.config(state=state)
        self.btn_csv.config(state=state)
        if disabled:
            self.btn_cancelar.config(state="disabled")

    def _show_result(self, conf: float, frames: int):
        self.result_var.set(f"Confiabilidad: {conf:.2f}/100")
        self.frames_var.set(f"Frames muestreados: {frames}")
        img_path = self.ok_img_path if conf >= 50.0 else self.bad_img_path
        try:
            pil = PILImage.open(img_path).resize((130, 130))
            self._img_tk = ImageTk.PhotoImage(pil)
            self.img_label.config(image=self._img_tk, text="")
        except Exception as e:
            self._img_tk = None
            self.img_label.config(image="", text=f"[No se pudo cargar imagen]\n{e}")

    def _require_video(self) -> str | None:
        ruta = self.video_path.get().strip()
        if not ruta:
            messagebox.showerror("Error", "Selecciona un video primero."); return None
        if not os.path.exists(ruta):
            messagebox.showerror("Error", f"No existe el archivo:\n{ruta}"); return None
        return ruta

    # ---------- Acciones ----------
    def cargar_video(self):
        ruta = filedialog.askopenfilename(title="Selecciona video",
                                          filetypes=[("Videos", "*.mp4;*.avi;*.mov;*.mkv"), ("Todos", "*.*")])
        if ruta:
            self.video_path.set(ruta)
            self.status_var.set("Video cargado. Listo para evaluar/visualizar/exportar.")

    def cancelar(self):
        self._cancel_eval.set()
        self.status_var.set("Cancelando… (espera un momento)")
        self.btn_cancelar.config(state="disabled")

    def evaluar_async(self):
        if BACKEND_ERROR or detect_face is None or classify_face is None:
            messagebox.showerror("Error", f"No se pudo usar el backend Haar.\n\n{BACKEND_ERROR}")
            return
        ruta = self._require_video()
        if not ruta: return
        if self._eval_thread and self._eval_thread.is_alive():
            messagebox.showinfo("En curso", "Ya hay una evaluación en curso."); return

        self._cancel_eval.clear()
        self._set_buttons_state(True)
        self.btn_cancelar.config(state="normal")
        self.status_var.set("Evaluando…")
        self.progress_var.set("")
        self._eval_thread = threading.Thread(target=self._evaluate_worker, args=(ruta, None), daemon=True)
        self._eval_thread.start()

    def exportar_csv_async(self):
        if BACKEND_ERROR or detect_face is None or classify_face is None:
            messagebox.showerror("Error", f"No se pudo usar el backend Haar.\n\n{BACKEND_ERROR}")
            return
        ruta = self._require_video()
        if not ruta: return
        out = filedialog.asksaveasfilename(title="Guardar CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not out: return
        if self._eval_thread and self._eval_thread.is_alive():
            messagebox.showinfo("En curso", "Ya hay una evaluación en curso."); return

        self._cancel_eval.clear()
        self._set_buttons_state(True)
        self.btn_cancelar.config(state="normal")
        self.status_var.set("Evaluando y exportando CSV…")
        self.progress_var.set("")
        self._eval_thread = threading.Thread(target=self._evaluate_worker, args=(ruta, out), daemon=True)
        self._eval_thread.start()

    def visualizar_async(self):
        if BACKEND_ERROR or visualizar_video is None:
            messagebox.showerror("Error", f"No se pudo usar el backend Haar.\n\n{BACKEND_ERROR}")
            return
        ruta = self._require_video()
        if not ruta: return
        self._set_buttons_state(True)
        self.status_var.set("Abriendo visualización (cierra con Q/Esc)…")
        threading.Thread(target=self._viz_worker, args=(ruta,), daemon=True).start()

    # ---------- Workers ----------
    def _viz_worker(self, ruta: str):
        try:
            visualizar_video(ruta)
        except Exception as e:
            self.after(0, lambda e=e: messagebox.showerror("Error en visualización", str(e)))
        finally:
            self.after(0, lambda: (self._set_buttons_state(False), self.status_var.set("Listo.")))

    def _evaluate_worker(self, ruta: str, csv_path: str | None):
        """
        Muestreo fijo 2 fps (0.5s). Si 'usar mirada' está activo y MediaPipe disponible,
        fusiona: S_final = S_ml_smooth * (alpha + (1-alpha)*F_smooth)
        """
        use_gaze = bool(self.use_gaze_var.get() and MP_AVAILABLE_FLAG and compute_frontal_score is not None)
        alpha = float(self.alpha_var.get())

        confs_final: list[float] = []
        sampled = 0
        ema_S = None
        ema_F = None

        try:
            cap = open_capture(ruta)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            duration_s = (total_frames / fps) if total_frames > 0 else 0.0

            # CSV
            if csv_path:
                f = open(csv_path, "w", newline="", encoding="utf-8")
                writer = csv.writer(f)
                if use_gaze:
                    writer.writerow(["second", "timestamp_s", "has_face", "yaw", "pitch", "F", "S_ml", "S_final"])
                else:
                    writer.writerow(["second", "timestamp_s", "has_face", "S_ml"])
            else:
                f = None; writer = None

            sec = 0.0
            last_update = time.time()

            while not self._cancel_eval.is_set():
                if duration_s and sec > duration_s:
                    break

                cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    break

                has_face = 0
                S_ml = 0.0
                S_final = 0.0
                yaw = None; pitch = None; F = None

                box = detect_face(frame)
                if box is not None:
                    has_face = 1
                    x1,y1,x2,y2 = box
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        S_ml = float(classify_face(crop))  # 0..100

                        # EMA de S_ml
                        ema_S = S_ml if ema_S is None else (EMA_S_ML* S_ml + (1-EMA_S_ML)* ema_S)

                        if use_gaze:
                            yaw, pitch = compute_frontal_score(frame, box)  # puede devolver (None,None)
                            F = frontal_score_from_yaw_pitch(yaw, pitch) if (yaw is not None) else None
                            if F is None:
                                # si no hay F este frame, mantener último EMA_F o asumir 1.0
                                F_inst = 1.0 if ema_F is None else ema_F
                            else:
                                ema_F = F if ema_F is None else (EMA_F* F + (1-EMA_F)* ema_F)
                                F_inst = ema_F
                            # fusión tardía
                            S_final = float(ema_S * (alpha + (1.0 - alpha) * F_inst))
                        else:
                            S_final = float(ema_S)

                        confs_final.append(S_final)

                # CSV por segundo
                if writer is not None:
                    if use_gaze:
                        writer.writerow([
                            int(sec), f"{sec:.3f}", has_face,
                            "" if yaw is None else f"{yaw:.3f}",
                            "" if pitch is None else f"{pitch:.3f}",
                            "" if F is None else f"{F:.3f}",
                            f"{S_ml:.3f}",
                            f"{S_final:.3f}" if has_face else ""
                        ])
                    else:
                        writer.writerow([int(sec), f"{sec:.3f}", has_face, f"{S_ml:.3f}"])

                sampled += 1
                sec += STEP_SECONDS

                now = time.time()
                if now - last_update > 0.25:
                    self.after(0, lambda s=sec: self.progress_var.set(f"Procesando t={s:.1f}s…"))
                    last_update = now

            cap.release()
            if csv_path and f:
                f.close()

            avg_final = (sum(confs_final) / len(confs_final)) if confs_final else 0.0
            self.after(0, lambda: self._show_result(avg_final, sampled))

            if csv_path:
                self.after(0, lambda p=csv_path: messagebox.showinfo("CSV guardado", f"Exportado a:\n{p}"))

        except Exception as e:
            self.after(0, lambda e=e: messagebox.showerror("Error al evaluar", str(e)))
        finally:
            self.after(0, lambda: (
                self._set_buttons_state(False),
                self.btn_cancelar.config(state="disabled"),
                self.status_var.set("Listo."),
                self.progress_var.set("" if not self._cancel_eval.is_set() else "Cancelado.")
            ))

# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    app = HaarApp()
    app.mainloop()
