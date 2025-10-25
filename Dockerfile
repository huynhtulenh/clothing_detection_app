FROM python:3.11-slim

# Cài đặt các dependencies hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file requirements và cài đặt thư viện Python
COPY backend/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Tải mô hình DeepFashion2 (hoặc sao chép nếu đã tải)
RUN wget -O backend/deepfashion2_yolov8s-seg.pt https://huggingface.co/Bingsu/adetailer/resolve/main/deepfashion2_yolov8s-seg.pt

# Sao chép toàn bộ mã nguồn
COPY . /app

# Expose cổng 8000
EXPOSE 8000

# Lệnh khởi động ứng dụng
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
