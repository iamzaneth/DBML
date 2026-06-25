# Huong dan chay trich xuat VSL landmarks

File script:

```powershell
python src\VSL\extract_vsl_landmarks.py
```

Input va output mac dinh:

- Input video: `data/interim/VSL_cropped/<label>/*.mp4`
- Output `.npz`: `data/processed/v2/VSL/<label>/*.npz`
- Output preview: `data/processed/v2/VSL_preview/<label>/*_preview.mp4`
- Schema: `data/processed/v2/VSL/feature_schema.json`

## Chay demo co preview

Lenh demo mau cho mot label:

```powershell
python src\VSL\extract_vsl_landmarks.py --labels <label> --all-frames --min-frames 60 --trim-action --trim-margin 12 --preview --overwrite
```

Vi du neu co label `xin_chao`:

```powershell
python src\VSL\extract_vsl_landmarks.py --labels xin_chao --all-frames --min-frames 60 --trim-action --trim-margin 12 --preview --overwrite
```

Lenh nay se:

- Chi chay label duoc chon.
- Xu ly toan bo video trong folder `data/interim/VSL_cropped/<label>`.
- Lay tat ca frame nguon, sau do trim doan khong co hanh dong.
- Giu toi thieu 60 time steps sau trim.
- Xuat preview co pose, hand va mouth landmarks.
- Ghi de file `.npz` va preview cu neu da ton tai.

## Chay theo worker

Dataset VSL duoc chia 10 worker theo thu tu folder da sort. Khac voi ASL, so label moi worker duoc tinh dong theo tong so label thuc te trong `data/interim/VSL_cropped`.

| Worker ID | Cach chay |
| --- | --- |
| 0 | Chay full tat ca label |
| 1..10 | Chia deu cac label da sort theo ten folder |

Worker ID ngoai `0..10` se bao loi. Worker `0` khong chia block, chay full.

Lenh mau cho tung thanh vien:

```powershell
python src\VSL\extract_vsl_landmarks.py --worker-id 1 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
```

Doi `--worker-id 1` thanh so cua tung nguoi:

```powershell
python src\VSL\extract_vsl_landmarks.py --worker-id 2 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
python src\VSL\extract_vsl_landmarks.py --worker-id 3 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
python src\VSL\extract_vsl_landmarks.py --worker-id 4 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
python src\VSL\extract_vsl_landmarks.py --worker-id 5 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
python src\VSL\extract_vsl_landmarks.py --worker-id 6 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
python src\VSL\extract_vsl_landmarks.py --worker-id 7 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
python src\VSL\extract_vsl_landmarks.py --worker-id 8 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
python src\VSL\extract_vsl_landmarks.py --worker-id 9 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
python src\VSL\extract_vsl_landmarks.py --worker-id 10 --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
```

## Chay full mot may

```powershell
python src\VSL\extract_vsl_landmarks.py --full --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
```

## Giai thich tham so

`--labels <label>`

Chay mot hoac nhieu label cu the. Label co khoang trang can dat trong dau nhay kep:

```powershell
python src\VSL\extract_vsl_landmarks.py --labels "xin chao" cam_on --all-frames --min-frames 60 --trim-action --trim-margin 12 --skip-existing
```

`--full`

Chay tat ca label. Neu dung `--worker-id` thi khong can them `--full`.

`--worker-id N`

Chia viec theo worker. `N=0` chay full, `N=1..10` chia deu cac label VSL da sort.

`--all-frames`

Lay tat ca frame nguon truoc khi trim. Cach nay giu nhieu thong tin chuyen dong hon so voi ep so frame ngay tu dau.

`--min-frames 60`

Sau khi trim, neu chuoi con duoi 60 time steps thi script se sample/duplicate deu len 60. Neu sau trim tren 60 thi giu nguyen, tru khi co `--max-frames`.

`--target-frames 60`

Dung khi muon output moi video co dung 60 time steps. Tham so nay loai tru voi `--all-frames`.

`--max-frames N`

Dat tran so frame neu video qua dai. Khi dung voi `--all-frames`, script lay toi da `N` frame va sau trim cung khong vuot qua `N`.

`--trim-action`

Cat bo doan dau/cuoi khong co hanh dong ro rang truoc khi noi suy tay. Nen bat khi chay dataset.

`--trim-margin 12`

Giu them 12 frame truoc/sau bien hanh dong sau khi trim. Tang gia tri nay neu preview bi cat qua sat hanh dong.

`--no-trim-action`

Tat trim de debug hoac so sanh.

`--stabilize-hand-side`

Bat sua loi MediaPipe nhay nhan trai/phai voi clip mot tay. Mac dinh dang bat.

`--no-stabilize-hand-side`

Tat sua nhan trai/phai neu gap label dung hai tay that su nhung bi gop nham.

`--hand-side-minority-ratio 0.20`

Nguong de gop side phu vao side chinh. Gia tri nho hon se chat hon, it gop hon.

`--preview`

Xuat video preview de kiem tra pose, hands va mouth sau buoc trim/resample. Rieng mouth duoc ve bang toa do raw cua 12 diem de overlay dung len frame goc; file `.npz` luu mouth da duoc normalize.

`--skip-existing`

Bo qua file `.npz` da ton tai. Nen dung khi chay worker/full de co the resume.

`--overwrite`

Ghi de file `.npz` va preview cu. Nen dung khi demo hoac khi muon tao lai label.

## Cau truc `.npz` hien tai

Moi file `.npz` luu cac key chinh:

- `pose`: `(T, 330)`
- `hands`: `(T, 252)`
- `face`: `(T, 52)` blendshape scores
- `mouth`: `(T, 36)` 12 mouth landmarks x `[x, y, z]`, da neo tai tam hai khoe mieng va scale theo do rong mieng
- `valid_mask`: `(T, 4)` voi cot `pose`, `left_hand`, `right_hand`, `face`

Tong feature dim khi train:

```python
x = np.concatenate([pose, hands, face, mouth], axis=1)
```

Hien tai:

```text
330 + 252 + 52 + 36 = 670
```
