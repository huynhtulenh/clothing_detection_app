# Hướng Dẫn Cài Đặt Ứng Dụng Phát Hiện Trang Phục

## 🧩 Bước 1: Cài đặt Dependencies

Mở terminal trong thư mục `backend` và chạy lệnh sau để cài đặt các thư
viện Python:

``` bash
pip install -r requirements.txt
```

## 🧠 Bước 2: Tải Mô hình

Tải file mô hình DeepFashion2 Segmentation và đặt nó vào thư mục
`backend`:

``` bash
wget -O backend/deepfashion2_yolov8s-seg.pt https://huggingface.co/Bingsu/adetailer/resolve/main/deepfashion2_yolov8s-seg.pt
```

## ⚙️ Bước 3: Chạy Backend API

Mở terminal trong thư mục `backend` và chạy server Uvicorn:

``` bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API sẽ chạy tại <http://0.0.0.0:8000>.

## 🌐 Bước 4: Chạy Frontend

Mở terminal trong thư mục `frontend` và chạy một HTTP server đơn giản:

``` bash
python -m http.server 8001
```

Frontend sẽ chạy tại <http://0.0.0.0:8001>.

## 🚀 Bước 5: Sử dụng Ứng dụng

1.  Mở trình duyệt và truy cập <http://localhost:8001>
2.  Nhấn "Chọn Ảnh" hoặc kéo thả ảnh để bắt đầu sử dụng ứng dụng.
