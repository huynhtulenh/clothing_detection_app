import io
import cv2 # OpenCV lib để đọc, vẽ trên ảnh
import numpy as np
import json # Dùng để chuyển đổi dữ liệu giữa Python và định dạng JSON
import base64 # Dùng để mã hoá ảnh hoặc dữ liệu nhị phân thành chuỗi ký tự (text) — dễ truyền qua API hoặc lưu trong JSON
from fastapi import FastAPI, File, UploadFile, HTTPException # API server
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO # Trong project DeepFashion2, YOLO sẽ phát hiện từng loại quần áo... trên ảnh
from typing import List, Dict, Any

# 1. Khởi tạo FastAPI app
app = FastAPI(
    title="Clothing Segmentation API",
    description="API nhận ảnh, phát hiện và phân đoạn trang phục chi tiết(quần, áo,..) bằng mô hình DeepFashion2.",
    version="1.0.0"
)

# Cấu hình CORS
origins = ["*"] # Cho phép tất cả các nguồn truy cập API
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Định nghĩa các lớp trang phục/phụ kiện và màu sắc tương ứng
# Các lớp này dựa trên mô hình DeepFashion2
CLOTHING_COLOR_MAP = {
    # Các lớp Áo
    "short_sleeved_shirt": (255, 0, 0), # Áo sơ mi ngắn tay - Màu đỏ
    "long_sleeved_shirt": (0, 255, 0),  # Áo sơ mi dài tay - Màu xanh lá
    "short_sleeved_outwear": (0,165,255), # Áo khoác ngắn tay - Màu cam
    "long_sleeved_outwear": (255,0,255),  # Áo khoác dài tay - Màu tím
    "vest": (0,255,255),                 # Áo vest - Màu vàng
    "sling": (128,0,128),                # Áo hai dây/Áo yếm - Màu tím đậm
    # Các lớp Quần/Váy
    "shorts": (0,0,255),                 # Quần ngắn - Màu xanh dương
    "trousers": (255,255,0),             # Quần dài - Màu vàng
    "skirt": (0,128,255),                # Váy - Màu xanh da trời
    "short_sleeved_dress": (128,128,0),  # Váy ngắn tay - Màu xanh rêu
    "long_sleeved_dress": (128,0,0),     # Váy dài tay - Màu nâu đỏ
    "vest_dress": (0,128,128),           # Váy vest - Màu xanh lục đậm
    "sling_dress": (128,128,128),        # Váy hai dây - Màu xám
}

# 3. Load mô hình YOLOv8 Segmentation DeepFashion2
MODEL_PATH = "deepfashion2_yolov8s-seg.pt" # Đường dẫn đến file mô hình đã huấn luyện
try:
    model = YOLO(MODEL_PATH)
    print(f"Mô hình YOLOv8s-seg DeepFashion2 đã được tải thành công từ {MODEL_PATH}.")
except Exception as e:
    print(f"Lỗi khi tải mô hình YOLOv8s-seg DeepFashion2: {e}")
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
@app.get("/")
async def root():
    return {"message": "Clothing Segmentation API is running. Use /upload to submit images."}

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

# ==================== SKETCH GENERATOR FUNCTIONS ====================

def create_sketch_effect(
    img: np.ndarray,
    edge_brightness: float = 10.0,
    edge_contrast: float = 1.2,
    sketch_brightness: float = 15.0,
    sketch_contrast: float = 1.0,
    noise_reduction: bool = False
) -> np.ndarray:
    """
    Create hand-drawn sketch effect from image using edge detection and image processing.
    
    Args:
        img: Input image (BGR format)
        edge_brightness: Brightness adjustment for edges (0-50)
        edge_contrast: Contrast adjustment for edges (0.5-3.0)
        sketch_brightness: Overall sketch brightness (0-50)
        sketch_contrast: Overall sketch contrast (0.5-3.0)
        noise_reduction: Apply Gaussian blur to reduce noise
    
    Returns:
        Sketch image (BGR format)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply noise reduction if enabled
    if noise_reduction:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Invert the grayscale image
    inverted = 255 - gray
    
    # Apply Gaussian blur to inverted image
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    
    # Invert the blurred image
    inverted_blur = 255 - blurred
    
    # Create sketch using divide blend mode
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    
    # Apply edge detection using Canny
    edges = cv2.Canny(gray, 50, 150)
    
    # Adjust edge brightness
    edges = cv2.convertScaleAbs(edges, alpha=1.0, beta=edge_brightness)
    
    # Adjust edge contrast
    edges = cv2.convertScaleAbs(edges, alpha=edge_contrast, beta=0)
    
    # Combine sketch with edges
    sketch = cv2.bitwise_or(sketch, edges)
    
    # Adjust overall sketch brightness
    sketch = cv2.convertScaleAbs(sketch, alpha=1.0, beta=sketch_brightness)
    
    # Adjust overall sketch contrast
    sketch = cv2.convertScaleAbs(sketch, alpha=sketch_contrast, beta=0)
    
    # Convert back to BGR
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    return sketch_bgr


def remove_background_threshold(img: np.ndarray) -> np.ndarray:
    """
    Remove background using simple threshold method (fastest).
    
    Args:
        img: Input image (BGR format)
    
    Returns:
        Image with transparent background (BGRA format)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Create BGRA image
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    
    return bgra


def remove_background_grabcut(img: np.ndarray) -> np.ndarray:
    """
    Remove background using GrabCut algorithm (more accurate but slower).
    
    Args:
        img: Input image (BGR format)
    
    Returns:
        Image with transparent background (BGRA format)
    """
    # Create a mask with all pixels marked as probable background
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # Define a rectangle around the foreground (center region)
    h, w = img.shape[:2]
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))
    
    # Initialize background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create binary mask (0 and 2 are background, 1 and 3 are foreground)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    
    # Create BGRA image
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask2
    
    return bgra


def remove_background_edge_based(img: np.ndarray) -> np.ndarray:
    """
    Remove background using edge-based detection (good for sketches).
    
    Args:
        img: Input image (BGR format)
    
    Returns:
        Image with transparent background (BGRA format)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 30, 100)
    
    # Dilate edges to create a mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=2)
    
    # Invert mask (we want to keep edges, not remove them)
    mask = cv2.bitwise_not(mask)
    
    # Apply threshold to clean up
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    
    # Create BGRA image
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    
    return bgra


def create_edge_image(
    img: np.ndarray,
    edge_brightness: float = 10.0,
    edge_contrast: float = 1.2,
    noise_reduction: bool = False
) -> np.ndarray:
    """
    Create edge detection image (white edges on black background).
    
    Args:
        img: Input image (BGR format)
        edge_brightness: Edge brightness adjustment
        edge_contrast: Edge contrast adjustment
        noise_reduction: Apply noise reduction
    
    Returns:
        Edge image (BGR format with black background)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply noise reduction if enabled
    if noise_reduction:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Adjust edge brightness
    edges = cv2.convertScaleAbs(edges, alpha=1.0, beta=edge_brightness)
    
    # Adjust edge contrast
    edges = cv2.convertScaleAbs(edges, alpha=edge_contrast, beta=0)
    
    # Convert to BGR (white edges on black background)
    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return edge_bgr


def detect_contours_info(img: np.ndarray, min_contour_area: int = 50) -> Dict[str, Any]:
    """
    Detect contours and return information for SVG conversion.
    
    Args:
        img: Input image (BGR or Grayscale)
        min_contour_area: Minimum area of contour to keep (default: 50)
    
    Returns:
        Dictionary with contour information
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Get image dimensions first
    height, width = gray.shape
    image_area = width * height
    
    # Threshold to binary (white lines on black background)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Invert if needed (we want black lines on white background for better contour detection)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    
    # Filter contours
    filtered_contours = []
    border_threshold = 10  # pixels from border
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip very small contours
        if area < min_contour_area:
            continue
        
        # Skip contours that are too large (likely the image border)
        if area > image_area * 0.95:
            continue
        
        # Check if contour is on the image border
        x, y, w, h = cv2.boundingRect(contour)
        is_border = (x <= border_threshold or 
                    y <= border_threshold or 
                    x + w >= width - border_threshold or 
                    y + h >= height - border_threshold)
        
        # Skip border contours that are too large
        if is_border and area > image_area * 0.5:
            continue
        
        filtered_contours.append(contour)
    
    # Count total points
    total_points = sum(len(contour) for contour in filtered_contours)
    
    print(f"Detected {len(contours)} contours, kept {len(filtered_contours)} after filtering")
    
    return {
        "num_contours": len(filtered_contours),
        "width": width,
        "height": height,
        "total_points": total_points,
        "contours": filtered_contours
    }


def generate_svg_from_contours(contours: list, width: int, height: int, method: str = "opencv") -> str:
    """
    Generate SVG string from contours.
    
    Args:
        contours: List of contours from OpenCV
        width: Image width
        height: Image height
        method: 'opencv' or 'potrace' style
    
    Returns:
        SVG string
    """
    svg_header = f'<?xml version="1.0" encoding="UTF-8"?>\n'
    svg_header += f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
    
    # Add white background
    svg_header += f'  <rect width="{width}" height="{height}" fill="white"/>\n'
    
    # Group paths for better organization
    svg_header += '  <g id="sketch-paths">\n'
    
    svg_paths = []
    path_count = 0
    
    # Sort contours by area (largest first) for better layering
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for idx, contour in enumerate(sorted_contours):
        if len(contour) < 3:  # Skip very small contours
            continue
        
        # Create path data with more efficient representation
        points = contour.reshape(-1, 2)  # Flatten contour points
        
        if len(points) < 2:
            continue
        
        # Use polyline simplification for smoother paths
        epsilon = 0.5  # Approximation accuracy
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 2:
            approx = contour
        
        approx_points = approx.reshape(-1, 2)
        
        # Start path
        path_data = f'M {approx_points[0][0]:.1f},{approx_points[0][1]:.1f} '
        
        # Add points with line commands
        for i in range(1, len(approx_points)):
            path_data += f'L {approx_points[i][0]:.1f},{approx_points[i][1]:.1f} '
        
        # Close path
        path_data += 'Z'
        
        # Different styles based on method
        if method == "potrace":
            # Potrace style: filled black paths with holes
            svg_paths.append(f'    <path id="path-{idx}" d="{path_data}" fill="black" stroke="none" fill-rule="evenodd"/>')
        else:
            # OpenCV style: stroked black paths with varying width based on area
            area = cv2.contourArea(contour)
            stroke_width = max(1, min(3, int(np.log10(area + 1))))  # Dynamic stroke width
            svg_paths.append(f'    <path id="path-{idx}" d="{path_data}" fill="none" stroke="black" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"/>')
        
        path_count += 1
    
    svg_footer = '  </g>\n</svg>'
    
    svg_content = svg_header + '\n'.join(svg_paths) + '\n' + svg_footer
    
    print(f"Generated SVG with {path_count} paths from {len(contours)} contours")
    
    return svg_content


def image_to_svg_bitmap(img: np.ndarray, width: int, height: int) -> str:
    """
    Convert image to SVG using bitmap embedding (preserves 100% detail).
    
    Args:
        img: Input image (grayscale or BGR)
        width: Target width
        height: Target height
    
    Returns:
        SVG string with embedded base64 image
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Encode image to PNG (lossless)
    is_success, buffer = cv2.imencode(".png", gray, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not is_success:
        raise Exception("Failed to encode image")
    
    # Convert to base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Create SVG with embedded image (this preserves ALL details from sketch)
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Hand-drawn Sketch (Bitmap Embedded)</title>
  <desc>This SVG contains the complete sketch as an embedded PNG image, preserving all details.</desc>
  <image width="{width}" height="{height}" xlink:href="data:image/png;base64,{img_base64}" image-rendering="optimizeQuality"/>
</svg>'''
    
    return svg


def image_to_svg_lines(img: np.ndarray, width: int, height: int, threshold: int = 200) -> str:
    """
    Convert sketch image to SVG by converting each dark pixel to a line/rect.
    This preserves all details but creates larger file size.
    
    Args:
        img: Input sketch image (grayscale)
        width: Image width
        height: Image height
        threshold: Pixel brightness threshold (0-255), pixels darker than this become lines
    
    Returns:
        SVG string with individual lines for each dark pixel
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    svg_header = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>Hand-drawn Sketch (Pixel-based Vector)</title>
  <rect width="{width}" height="{height}" fill="white"/>
  <g id="sketch-pixels">
'''
    
    # Scan image and create tiny rectangles for dark pixels
    # Use a more efficient approach: scan by rows and create horizontal lines
    svg_lines = []
    line_count = 0
    
    for y in range(height):
        x_start = None
        for x in range(width):
            pixel_value = gray[y, x]
            
            if pixel_value < threshold:  # Dark pixel (part of sketch)
                if x_start is None:
                    x_start = x  # Start of line
            else:  # Light pixel (background)
                if x_start is not None:
                    # End of line, draw it
                    line_width = x - x_start
                    if line_width > 0:
                        # Solid black line (full opacity)
                        svg_lines.append(f'    <line x1="{x_start}" y1="{y}" x2="{x}" y2="{y}" stroke="black" stroke-width="1.5" stroke-linecap="round"/>')
                        line_count += 1
                    x_start = None
        
        # Handle line extending to edge
        if x_start is not None:
            line_width = width - x_start
            if line_width > 0:
                svg_lines.append(f'    <line x1="{x_start}" y1="{y}" x2="{width}" y2="{y}" stroke="black" stroke-width="1.5" stroke-linecap="round"/>')
                line_count += 1
    
    svg_footer = '''  </g>
</svg>'''
    
    print(f"Generated pixel-based SVG with {line_count} lines")
    
    # If too many lines (> 50000), return bitmap version instead
    if line_count > 50000:
        print(f"Too many lines ({line_count}), using bitmap embedding instead")
        return image_to_svg_bitmap(img, width, height)
    
    return svg_header + '\n'.join(svg_lines) + '\n' + svg_footer


@app.post("/sketch")
async def create_sketch(
    file: UploadFile = File(...),
    edge_brightness: float = 10.0,
    edge_contrast: float = 1.2,
    sketch_brightness: float = 15.0,
    sketch_contrast: float = 1.0,
    noise_reduction: bool = False,
    background_removal: bool = False,
    bg_removal_method: str = "threshold",
    enable_segmentation: bool = True  # NEW: Enable clothing detection
):
    """
    Create enhanced hand-drawn sketch from uploaded image with optional clothing segmentation.
    
    Args:
        file: Uploaded image file
        edge_brightness: Edge brightness adjustment (0-50)
        edge_contrast: Edge contrast adjustment (0.5-3.0)
        sketch_brightness: Sketch brightness adjustment (0-50)
        sketch_contrast: Sketch contrast adjustment (0.5-3.0)
        noise_reduction: Apply noise reduction
        background_removal: Remove background from sketch
        bg_removal_method: Background removal method ('threshold', 'grabcut', 'edge_based')
        enable_segmentation: Enable clothing detection and segmentation (polygon data)
    
    Returns:
        JSON with base64 encoded images (original, edge, sketch, segmented), contour info, and segmentation data
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_cv is None:
            raise HTTPException(status_code=400, detail="Cannot decode image.")

        # 1. Encode original image
        is_success, buffer_orig = cv2.imencode(".jpg", img_cv)
        if not is_success:
            raise HTTPException(status_code=500, detail="Error encoding original image.")
        original_base64 = base64.b64encode(buffer_orig).decode("utf-8")

        # 1.5 Clothing Detection & Segmentation (if enabled)
        segmented_base64 = None
        segmentation_data = []
        if enable_segmentation:
            try:
                print("Running clothing segmentation detection...")
                results = model(img_cv, classes=target_class_ids, verbose=False)
                segmented_img, segmentation_data = draw_segmentation(img_cv, results, target_class_names, CLOTHING_COLOR_MAP)
                
                is_success_seg, buffer_seg = cv2.imencode(".jpg", segmented_img)
                if is_success_seg:
                    segmented_base64 = base64.b64encode(buffer_seg).decode("utf-8")
                    print(f"Detected {len(segmentation_data)} clothing items")
            except Exception as e:
                print(f"Warning: Segmentation failed: {e}")
                # Continue without segmentation if it fails

        # 2. Create edge detection image
        edge_img = create_edge_image(
            img_cv,
            edge_brightness=edge_brightness,
            edge_contrast=edge_contrast,
            noise_reduction=noise_reduction
        )
        is_success, buffer_edge = cv2.imencode(".jpg", edge_img)
        if not is_success:
            raise HTTPException(status_code=500, detail="Error encoding edge image.")
        edge_base64 = base64.b64encode(buffer_edge).decode("utf-8")

        # 3. Create sketch effect
        sketch_img = create_sketch_effect(
            img_cv,
            edge_brightness=edge_brightness,
            edge_contrast=edge_contrast,
            sketch_brightness=sketch_brightness,
            sketch_contrast=sketch_contrast,
            noise_reduction=noise_reduction
        )

        # Apply background removal if enabled
        if background_removal:
            if bg_removal_method == "threshold":
                sketch_img = remove_background_threshold(sketch_img)
            elif bg_removal_method == "grabcut":
                sketch_img = remove_background_grabcut(sketch_img)
            elif bg_removal_method == "edge_based":
                sketch_img = remove_background_edge_based(sketch_img)
            else:
                raise HTTPException(status_code=400, detail=f"Invalid background removal method: {bg_removal_method}")
            
            # Encode as PNG to preserve transparency
            is_success, buffer_sketch = cv2.imencode(".png", sketch_img)
        else:
            # Encode as JPG
            is_success, buffer_sketch = cv2.imencode(".jpg", sketch_img)
        
        if not is_success:
            raise HTTPException(status_code=500, detail="Error encoding sketch image.")
        
        sketch_base64 = base64.b64encode(buffer_sketch).decode("utf-8")

        # 4. Detect contours for vector conversion
        # Use sketch image instead of edge image for better contour detection
        gray_sketch = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2GRAY) if len(sketch_img.shape) == 3 else sketch_img
        
        # Get image dimensions
        h, w = gray_sketch.shape if len(gray_sketch.shape) == 2 else gray_sketch.shape[:2]
        
        # Detect contours from sketch (has more details than edge)
        contour_info = detect_contours_info(gray_sketch, min_contour_area=100)
        
        # If no contours found, try with edge image
        if contour_info["num_contours"] == 0:
            print("No contours in sketch, trying edge image...")
            gray_edge = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
            contour_info = detect_contours_info(gray_edge, min_contour_area=50)
        
        print(f"Final contour count: {contour_info['num_contours']}")
        
        # Generate SVG OpenCV style - PIXEL-BASED (preserves ALL details from sketch)
        print("Generating pixel-based SVG (OpenCV)...")
        svg_opencv = image_to_svg_lines(gray_sketch, w, h, threshold=200)
        svg_opencv_base64 = base64.b64encode(svg_opencv.encode('utf-8')).decode('utf-8')
        
        # Generate SVG Potrace style (contour-based filled paths)
        print("Generating contour-based SVG (Potrace)...")
        svg_potrace = generate_svg_from_contours(
            contour_info["contours"],
            contour_info["width"],
            contour_info["height"],
            method="potrace"
        )
        svg_potrace_base64 = base64.b64encode(svg_potrace.encode('utf-8')).decode('utf-8')
        
        # Generate bitmap-based SVG (embedded PNG - perfect quality)
        print("Generating bitmap SVG...")
        svg_bitmap = image_to_svg_bitmap(gray_sketch, w, h)
        svg_bitmap_base64 = base64.b64encode(svg_bitmap.encode('utf-8')).decode('utf-8')

        return JSONResponse(content={
            "status": "success",
            "message": "Sketch created successfully",
            "original_image_base64": original_base64,
            "edge_image_base64": edge_base64,
            "sketch_image_base64": sketch_base64,
            "segmented_image_base64": segmented_base64,  # NEW: Segmented image with polygon overlay
            "segmentation_results": segmentation_data,    # NEW: Polygon coordinates and labels
            "contour_info": {
                "num_contours": contour_info["num_contours"],
                "width": contour_info["width"],
                "height": contour_info["height"],
                "total_points": contour_info["total_points"]
            },
            "svg_opencv_base64": svg_opencv_base64,
            "svg_potrace_base64": svg_potrace_base64,
            "svg_bitmap_base64": svg_bitmap_base64,
            "settings": {
                "edge_brightness": edge_brightness,
                "edge_contrast": edge_contrast,
                "sketch_brightness": sketch_brightness,
                "sketch_contrast": sketch_contrast,
                "noise_reduction": noise_reduction,
                "background_removal": background_removal,
                "bg_removal_method": bg_removal_method if background_removal else None,
                "enable_segmentation": enable_segmentation
            }
        })

    except Exception as e:
        print(f"Error processing sketch: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating sketch: {e}")