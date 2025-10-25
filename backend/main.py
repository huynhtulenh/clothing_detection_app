import io
import cv2
import numpy as np
import json
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from typing import List, Dict, Any
from starlette.staticfiles import StaticFiles
from pathlib import Path

# 1. Khởi tạo FastAPI App
app = FastAPI(
    title="Clothing Segmentation API (DeepFashion2 - Final)",
    description="API nhận ảnh, phát hiện và phân đoạn trang phục chi tiết (quần, áo, giày, váy) bằng mô hình DeepFashion2.",
    version="2.1.0"
)

# Cấu hình CORS
origins = ["*"] # Cho phép tất cả các nguồn
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Định nghĩa các lớp trang phục/phụ kiện và màu sắc tương ứng (BGR)
CLOTHING_COLOR_MAP = {
    "short_sleeved_shirt": (255, 0, 0),      # Xanh Dương - Áo cộc tay
    "long_sleeved_shirt": (0, 255, 0),       # Xanh Lá - Áo dài tay
    "short_sleeved_outwear": (0, 165, 255),  # Cam - Áo khoác cộc tay
    "long_sleeved_outwear": (255, 0, 255),   # Tím - Áo khoác dài tay
    "vest": (0, 255, 255),                   # Xanh Lục Lam - Áo vest/Gile
    "sling": (128, 0, 128),                  # Tím đậm - Áo hai dây/Áo yếm
    "shorts": (0, 0, 255),                   # Đỏ - Quần ngắn
    "trousers": (255, 255, 0),               # Vàng - Quần dài
    "skirt": (0, 128, 255),                  # Cam đậm - Chân váy
    "short_sleeved_dress": (128, 128, 0),    # Xanh xám - Váy cộc tay
    "long_sleeved_dress": (128, 0, 0),       # Xanh đậm - Váy dài tay
    "vest_dress": (0, 128, 128),             # Xanh lục lam đậm - Váy vest
    "sling_dress": (128, 128, 128),          # Xám - Váy hai dây
}

# 3. Load mô hình YOLOv8 Segmentation DeepFashion2
# LƯU Ý: Khi deploy, file mô hình sẽ nằm trong thư mục /app/backend/
MODEL_PATH = Path("backend/deepfashion2_yolov8s-seg.pt")
try:
    model = YOLO(str(MODEL_PATH))
    print(f"Mô hình YOLOv8s-seg DeepFashion2 đã được tải thành công từ {MODEL_PATH}.")
except Exception as e:
    print(f"Lỗi khi tải mô hình YOLOv8s-seg DeepFashion2: {e}")
    # Nếu mô hình không tồn tại (chưa được tải trong Dockerfile), sẽ dùng mô hình mặc định
    if not MODEL_PATH.exists():
        print("Cảnh báo: Không tìm thấy mô hình DeepFashion2. Đang sử dụng YOLOv8n-seg mặc định.")
        model = YOLO("yolov8n-seg.pt") # Fallback to default model
    else:
        raise RuntimeError(f"Không thể tải mô hình: {e}")

# Lấy ID của các lớp cần lọc
target_class_names = list(CLOTHING_COLOR_MAP.keys())
target_class_ids = [k for k, v in model.names.items() if v in target_class_names]


# 4. Hàm vẽ polygon lên ảnh
def draw_segmentation(img: np.ndarray, results: Any, target_class_names: List[str], color_map: Dict[str, tuple]) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    """Vẽ polygon lên ảnh với màu sắc tùy chỉnh và chỉ giữ lại các lớp trang phục."""
    
    draw_img = img.copy()
    h, w, _ = draw_img.shape
    
    masks = results[0].masks
    boxes = results[0].boxes
    
    if masks is None:
        return draw_img, []

    segmentation_data: List[Dict[str, Any]] = []

    for i, mask_xy in enumerate(masks.xyn):
        cls = int(boxes.cls[i])
        name = results[0].names[cls]
        conf = float(boxes.conf[i])
        
        if name not in target_class_names:
            continue

        color = color_map.get(name, (255, 255, 255))
        
        mask_pixel = masks.data[i].cpu().numpy().astype(np.uint8)
        mask_pixel = cv2.resize(mask_pixel, (w, h), interpolation=cv2.INTER_NEAREST)
        
        overlay = np.zeros_like(draw_img, dtype=np.uint8)
        overlay[mask_pixel == 1] = color
        
        alpha = 0.4
        draw_img = cv2.addWeighted(draw_img, 1, overlay, alpha, 0)
        
        contours, _ = cv2.findContours(mask_pixel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(draw_img, contours, -1, color, 2)
        
        polygon_coords = (mask_xy * np.array([w, h])).astype(int).tolist()
        
        bbox = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = bbox
        
        label_text = f"{name} {conf:.2f}"
        
        cv2.putText(draw_img, label_text, (x1, y1 - 10 if y1 > 20 else y1 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        segmentation_data.append({
            "id": i,
            "label": name,
            "confidence": round(conf, 4),
            "polygon": polygon_coords
        })

    return draw_img, segmentation_data


# 5. Định nghĩa Endpoint
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Nhận ảnh, thực hiện phân đoạn (segmentation) với lọc trang phục, trả về ảnh đã vẽ polygon và JSON tọa độ viền.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File phải là định dạng ảnh.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_cv is None:
            raise HTTPException(status_code=400, detail="Không thể giải mã ảnh.")

        # Thực hiện dự đoán với YOLOv8 Segmentation, chỉ lọc các lớp đã định nghĩa
        results = model(img_cv, classes=target_class_ids, verbose=False)
        
        processed_img, segmentation_data = draw_segmentation(img_cv, results, target_class_names, CLOTHING_COLOR_MAP)

        is_success, buffer = cv2.imencode(".jpg", processed_img)
        if not is_success:
            raise HTTPException(status_code=500, detail="Lỗi khi mã hóa ảnh kết quả.")
        
        img_bytes = io.BytesIO(buffer)

        encoded_image = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "status": "success",
            "message": f"Đã phát hiện {len(segmentation_data)} đối tượng trang phục/phụ kiện chi tiết.",
            "processed_image_base64": encoded_image,
            "segmentation_results": segmentation_data
        })

    except Exception as e:
        print(f"Lỗi xử lý: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {e}")

# 6. Phục vụ Frontend
# Đảm bảo thư mục frontend tồn tại và có file index.html
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend_static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Phục vụ file index.html từ thư mục frontend."""
    try:
        with open("frontend/index.html", "r") as f:
            # Thay thế URL API trong index.html để sử dụng đường dẫn tương đối /upload
            content = f.read()
            # Tìm và thay thế URL API trong JavaScript
            # Vì trong môi trường deploy, FE và BE chạy trên cùng 1 host, ta chỉ cần dùng /upload
            content = content.replace("const BACKEND_URL = 'http://localhost:8000/upload';", "const BACKEND_URL = '/upload';")
            return content
    except FileNotFoundError:
        return HTMLResponse("<h1>Lỗi: Không tìm thấy file Frontend (index.html).</h1>", status_code=404)