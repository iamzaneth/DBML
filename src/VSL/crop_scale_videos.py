import os
import cv2
from pathlib import Path
import concurrent.futures
import time

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "interim" / "VSL"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "VSL_cropped"

# ==========================================
# CẤU HÌNH CROP & SCALE
# ==========================================
CROP_X_START = 200
CROP_X_END = 1100
SCALE_PERCENT = 0.5  # Thu nhỏ 50%

def process_video(video_path: Path):
    """
    Xử lý 1 video: Crop -> Scale -> Lưu ra file mới giữ nguyên cấu trúc thư mục.
    """
    try:
        # Lấy đường dẫn tương đối (chứa tên label và tên video)
        # Ví dụ: ai/W00004B.mp4
        rel_path = video_path.relative_to(INPUT_DIR)
        out_path = OUTPUT_DIR / rel_path
        
        # Tạo thư mục label tương ứng ở OUTPUT_DIR nếu chưa có
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Bỏ qua nếu video đã được xử lý rồi (tiện cho việc chạy tiếp nếu bị đứt quãng)
        if out_path.exists():
            return True, f"Skipped (Exists): {rel_path}"
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, f"Failed to open: {rel_path}"
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0
        
        # Kích thước đầu ra
        out_width = int((CROP_X_END - CROP_X_START) * SCALE_PERCENT) # 450
        out_height = int(720 * SCALE_PERCENT) # 360
        
        # Định dạng codec mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (out_width, out_height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Cắt 2 bên (lấy từ pixel 200 đến 1100 chiều ngang, giữ nguyên chiều dọc)
            cropped = frame[:, CROP_X_START:CROP_X_END]
            
            # Thu nhỏ lại
            resized = cv2.resize(cropped, (out_width, out_height), interpolation=cv2.INTER_AREA)
            
            # Ghi frame
            out.write(resized)
            
        cap.release()
        out.release()
        return True, f"Success: {rel_path}"
        
    except Exception as e:
        return False, f"Error at {video_path.name}: {str(e)}"

def main():
    if not INPUT_DIR.exists():
        print(f"Error: Thư mục đầu vào không tồn tại - {INPUT_DIR}")
        return

    # Quét toàn bộ file .mp4 trong thư mục input và các thư mục con
    print(f"Đang quét tìm video trong {INPUT_DIR}...")
    video_files = list(INPUT_DIR.rglob("*.mp4"))
    total_videos = len(video_files)
    
    if total_videos == 0:
        print("Không tìm thấy video nào để xử lý.")
        return
        
    print(f"Tìm thấy tổng cộng {total_videos} videos.")
    print(f"Thư mục đầu ra: {OUTPUT_DIR}")
    print("-" * 50)
    
    start_time = time.time()
    success_count = 0
    fail_count = 0

    # Sử dụng ProcessPoolExecutor để chạy song song đa luồng (với số luồng = số nhân CPU)
    # Tốc độ sẽ tăng gấp nhiều lần so với chạy tuần tự.
    max_workers = os.cpu_count() or 4
    print(f"Bắt đầu xử lý đa luồng với {max_workers} CPU cores...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Đẩy tất cả task vào pool
        futures = {executor.submit(process_video, vf): vf for vf in video_files}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                
            # In tiến trình mỗi 50 video
            if i % 50 == 0 or i == total_videos:
                print(f"Progress: [{i}/{total_videos}] - Success: {success_count}, Fail: {fail_count}")

    elapsed_time = time.time() - start_time
    print("-" * 50)
    print(f"HOÀN THÀNH!")
    print(f"Tổng thời gian: {elapsed_time:.2f} giây")
    print(f"Xử lý thành công: {success_count}/{total_videos}")
    print(f"Lỗi: {fail_count}")

if __name__ == "__main__":
    main()
