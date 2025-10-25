echo "FROM python:3.11-slim" > /home/ubuntu/clothing_detection_app/Dockerfile
echo "" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "# Cài đặt các dependencies hệ thống cần thiết cho OpenCV" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "RUN apt-get update && apt-get install -y \\" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "    libgl1-mesa-glx \\" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "    libsm6 \\" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "    libxext6 \\" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "    libxrender1 \\" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "    wget \\" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "    && rm -rf /var/lib/apt/lists/*" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "# Thiết lập thư mục làm việc" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "WORKDIR /app" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "# Sao chép file requirements và cài đặt thư viện Python" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "COPY backend/requirements.txt /app/" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "RUN pip install --no-cache-dir -r requirements.txt" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "# Tải mô hình DeepFashion2 (hoặc sao chép nếu đã tải)" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "RUN wget -O backend/deepfashion2_yolov8s-seg.pt https://huggingface.co/Bingsu/adetailer/resolve/main/deepfashion2_yolov8s-seg.pt" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "# Sao chép toàn bộ mã nguồn" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "COPY . /app" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "# Expose cổng 8000" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "EXPOSE 8000" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "# Lệnh khởi động ứng dụng" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "# Uvicorn sẽ phục vụ cả API và file index.html (sẽ sửa main.py để phục vụ file tĩnh )" >> /home/ubuntu/clothing_detection_app/Dockerfile
echo "CMD [\"uvicorn\", \"backend.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]" >> /home/ubuntu/clothing_detection_app/Dockerfile
