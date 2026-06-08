# Nhận diện Ngôn ngữ Ký hiệu Việt Nam (VSL) Real-time

Dự án nhận diện và dịch Ngôn ngữ Ký hiệu Việt Nam theo thời gian thực sử dụng TensorFlow, MediaPipe Holistic và dataset VOYA_VSL. 

## Tính năng
- **Real-time Inference**: Trích xuất landmark qua webcam và dịch trực tiếp sang Tiếng Việt.
- **Pipeline toàn diện**: Từ tải dataset, tiền xử lý, augmentation cho landmark data đến training và evaluation.
- **3 Kiến trúc Model**: 
  - `lstm`: Bidirectional LSTM baseline (nhẹ, nhanh).
  - `transformer`: Kiến trúc self-attention cho data dạng chuỗi.
  - `hybrid`: CNN + BiLSTM + Attention (khuyến nghị, model nhẹ nhất ~5.6MB).
- **Tối ưu hiệu năng**: Mixed precision training, cấu hình cho RTX 4050 Laptop GPU.

## Cài đặt

1. Tạo và kích hoạt môi trường ảo (khuyến nghị Python 3.10+):
```bash
python -m venv .venv
# Kích hoạt trên Windows:
.venv\Scripts\activate
# Kích hoạt trên Linux/Mac:
source .venv/bin/activate
```

2. Cài đặt các thư viện yêu cầu:
```bash
pip install -r requirements.txt
```

## Luồng công việc (Workflow)

### 1. Tải và chuẩn bị dữ liệu
Tải dataset VOYA_VSL (HuggingFace) và lưu dưới dạng các file `.npz`:
```bash
python -m data.download_dataset --output_dir data/raw
```
*(Lệnh này cũng sẽ tự động trích xuất các labels thành file `labels.json` trong thư mục gốc)*

### 2. Khám phá dữ liệu (Tùy chọn)
Kiểm tra tính toàn vẹn và in các thống kê chi tiết của dataset:
```bash
python -m data.download_dataset --explore
```

### 3. Training
Train model (mặc định sẽ dùng cấu hình hybrid từ `config.yaml`):
```bash
python train.py --config config/config.yaml
```

**Các tham số bổ sung:**
- Thay đổi model type: `--model_type lstm` hoặc `--model_type transformer`
- Thay đổi batch size/epochs: `--batch_size 64 --epochs 100`
- Tắt mixed precision: `--no_mixed_precision`
- Chạy trên GPU cụ thể: `--gpu 0`

Kết quả training (checkpoints, logs, learning curves, confusion matrix) sẽ được lưu trong thư mục `outputs/` (hoặc tuỳ chỉnh qua `--output_dir`).

### 4. Đánh giá (Evaluation)
Nếu chỉ muốn chạy đánh giá một checkpoint đã có:
```bash
python train.py --config config/config.yaml --evaluate_only --resume outputs/best_model.h5
```

### 5. Inference (Nhận diện thời gian thực)
Sử dụng webcam để nhận diện thời gian thực:
```bash
python infer.py --model outputs/best_model.h5 --mode webcam
```

**Các tuỳ chọn Inference:**
- Tắt vẽ landmark (giảm lag): `--no_landmarks`
- Mượt kết quả (smoothing, trung bình qua N frames): `--smoothing 5`
- Thay đổi số lượng kết quả hiển thị (Top-K): `--top_k 3`

## Cấu trúc Project
```
DBML/
├── config/              # Cấu hình siêu tham số (yaml) và loader
├── data/                # Tải dữ liệu, Dataset class, tiền xử lý, augmentations
├── models/              # Kiến trúc Keras: Hybrid, Transformer, LSTM
├── training/            # Loop training, callbacks, evaluation
├── inference/           # Real-time inference bằng MediaPipe
├── utils/               # Công cụ vẽ đồ thị, lưu logs
├── train.py             # Script chạy training chính
├── infer.py             # Script chạy webcam/inference chính
├── requirements.txt     # Danh sách thư viện Python
└── README.md            # Tài liệu
```

## Yêu cầu phần cứng
- RAM tối thiểu 8GB (đề nghị 16GB).
- NVIDIA GPU (khuyến nghị dòng RTX) để train nhanh hơn, ví dụ RTX 4050 hỗ trợ float16.
- CPU hiện đại đủ chạy ổn định inference MediaPipe ở 30FPS.
