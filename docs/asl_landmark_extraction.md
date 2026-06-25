# Hướng dẫn chạy trích xuất ASL landmarks

File script:

```powershell
python src\ASL\extract_asl_landmarks.py
```

Input và output mặc định:

- Input video: `data/interim/ASL/<label>/*.mp4`
- Output `.npz`: `data/processed/v2/ASL/<label>/*.npz`
- Output preview: `data/processed/v2/ASL_preview/<label>/*_preview.mp4`
- Schema: `data/processed/v2/ASL/feature_schema.json`

## Chạy demo có preview

Lệnh demo mẫu cho label `a`:

```powershell
python src\ASL\extract_asl_landmarks.py --labels a --all-frames --min-frames 60 --trim-action --trim-margin 12 --preview --overwrite
```

Lệnh này sẽ:

- Chỉ chạy label `a`.
- Xử lý toàn bộ video trong folder `data/interim/ASL/a`.
- Lấy tất cả frame nguồn, sau đó trim đoạn không có hành động.
- Giữ tối thiểu 60 time steps sau trim.
- Xuất preview có pose, hand và mouth landmarks.
- Ghi đè file `.npz` và preview cũ nếu đã tồn tại.

## Chạy theo worker

Dataset ASL hiện tại được chia 10 worker, mỗi worker xử lý 200 label theo thứ tự folder đã sort.

| Worker ID | Label index |
| --- | --- |
| 0 | Chạy full tất cả label |
| 1 | 0-199 |
| 2 | 200-399 |
| 3 | 400-599 |
| 4 | 600-799 |
| 5 | 800-999 |
| 6 | 1000-1199 |
| 7 | 1200-1399 |
| 8 | 1400-1599 |
| 9 | 1600-1799 |
| 10 | 1800-1999 |

Worker ID ngoài `0..10` sẽ báo lỗi. Worker `0` không chia block, chạy full.

Lệnh mẫu cho từng thành viên:

```powershell
python src\ASL\extract_asl_landmarks.py --worker-id 1 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
```

Đổi `--worker-id 1` thành số của từng người:

```powershell
python src\ASL\extract_asl_landmarks.py --worker-id 2 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
python src\ASL\extract_asl_landmarks.py --worker-id 3 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
python src\ASL\extract_asl_landmarks.py --worker-id 4 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
python src\ASL\extract_asl_landmarks.py --worker-id 5 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
python src\ASL\extract_asl_landmarks.py --worker-id 6 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
python src\ASL\extract_asl_landmarks.py --worker-id 7 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
python src\ASL\extract_asl_landmarks.py --worker-id 8 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
python src\ASL\extract_asl_landmarks.py --worker-id 9 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
python src\ASL\extract_asl_landmarks.py --worker-id 10 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
```

## Chạy full một máy

```powershell
python src\ASL\extract_asl_landmarks.py --full --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
```

## Giải thích tham số

`--labels a`

Chạy một hoặc nhiều label cụ thể. Label có khoảng trắng cần đặt trong dấu nháy kép:

```powershell
python src\ASL\extract_asl_landmarks.py --labels a "a lot" abdomen --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
```

`--full`

Chạy tất cả label. Nếu dùng `--worker-id` thì không cần thêm `--full`.

`--worker-id N`

Chia việc theo worker. `N=0` chạy full, `N=1..10` mỗi worker xử lý đúng 200 label.

`--all-frames`

Lấy tất cả frame nguồn trước khi trim. Cách này giữ nhiều thông tin chuyển động hơn so với ép số frame ngay từ đầu.

`--min-frames 60`

Sau khi trim, nếu chuỗi còn dưới 60 time steps thì script sẽ sample/duplicate đều lên 60. Nếu sau trim trên 60 thì giữ nguyên, trừ khi có `--max-frames`.

`--target-frames 60`

Dùng khi muốn output mỗi video có đúng 60 time steps. Tham số này loại trừ với `--all-frames`.

`--max-frames N`

Đặt trần số frame nếu video quá dài. Khi dùng với `--all-frames`, script lấy tối đa `N` frame và sau trim cũng không vượt quá `N`.

`--trim-action`

Cắt bỏ đoạn đầu/cuối không có hành động rõ ràng trước khi nội suy tay. Nên bật khi chạy dataset.

`--trim-margin 12`

Giữ thêm 12 frame trước/sau biên hành động sau khi trim. Tăng giá trị này nếu preview bị cắt quá sát hành động.

`--no-trim-action`

Tắt trim để debug hoặc so sánh.

`--stabilize-hand-side`

Bật sửa lỗi MediaPipe nhảy nhãn trái/phải với clip một tay. Mặc định đang bật.

`--no-stabilize-hand-side`

Tắt sửa nhãn trái/phải nếu gặp label dùng hai tay thật sự nhưng bị gộp nhầm.

`--hand-side-minority-ratio 0.20`

Ngưỡng để gộp side phụ vào side chính. Giá trị nhỏ hơn sẽ chặt hơn, ít gộp hơn.

`--preview`

Xuất video preview để kiểm tra pose, hands và mouth sau bước trim/resample. Riêng mouth được vẽ bằng tọa độ raw của 12 điểm để overlay đúng lên frame gốc; file `.npz` lưu mouth đã được normalize.

`--skip-existing`

Bỏ qua file `.npz` đã tồn tại. Nên dùng khi chạy worker/full để có thể resume.

`--overwrite`

Ghi đè file `.npz` và preview cũ. Nên dùng khi demo hoặc khi muốn tạo lại label.

## Cấu trúc `.npz` hiện tại

Mỗi file `.npz` lưu các key chính:

- `pose`: `(T, 330)`
- `hands`: `(T, 252)`
- `face`: `(T, 52)` blendshape scores
- `mouth`: `(T, 36)` 12 mouth landmarks x `[x, y, z]`, đã neo tại tâm hai khoé miệng và scale theo độ rộng miệng
- `valid_mask`: `(T, 4)` với cột `pose`, `left_hand`, `right_hand`, `face`

Tổng feature dim khi train:

```python
x = np.concatenate([pose, hands, face, mouth], axis=1)
```

Hiện tại:

```text
330 + 252 + 52 + 36 = 670
```
