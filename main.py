import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.onnx")
model = YOLO(MODEL_PATH)

# Map tên class về đúng 2 key bạn muốn
def to_key(label: str) -> str | None:
    s = str(label).lower().replace(" ", "").replace("-", "_")
    if "bo_ray" in s or "boray" in s:
        return "Bo_Ray"
    if "sau_khoang" in s or "saukhoang" in s:
        return "Sau_Khoang"
    return None

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.7,
):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Ảnh lỗi / không đọc được"}

    results = model.predict(source=img, conf=conf, iou=iou, verbose=False)
    r = results[0]

    counts = {"Bo_Ray": 0, "Sau_Khoang": 0}

    if r.boxes is not None and r.boxes.cls is not None:
        cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int).tolist()
        names = model.names  # dict: id -> name

        for cid in cls_ids:
            label = names.get(cid, str(cid))
            key = to_key(label)
            if key is not None:
                counts[key] += 1

    return counts