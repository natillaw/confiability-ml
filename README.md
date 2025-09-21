# 📊 Confiabilidad ML — OpenCV (Haar) + PyTorch

Proyecto para estimar una puntuación de “confiabilidad” en video a partir del rostro:
- Detección facial con **Haar Cascades (OpenCV)**.
- Clasificador **PyTorch** (CNN ligera) entrenado con frames `lie`/`truth`.
- **GUI en Tkinter** para evaluar, visualizar y exportar CSV.
- Señal opcional de mirada con **MediaPipe Face Mesh** (penaliza si no mira a cámara).

> **Nota:** El dataset usado NO es de mi autoría. Ver sección [Dataset](#-dataset).

---

## ✨ Resultados (ejemplo de mi entrenamiento)
- **Test Acc:** 97.54%  
- **Precisión / Recall / F1:** ~0.97 / ~0.97 / ~0.97  
*(tu resultado puede variar en función del split y los frames)*

---

## 🚀 Instalación rápida

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

