# Report 4.8 - Similarity Analysis between VSL and WLASL

## Pipeline
The pipeline reads available CSV outputs, validates merge keys, merges per-video feature tables, aggregates video features into gloss-level vectors, standardizes all gloss vectors with one StandardScaler, then computes cosine similarity for mapped VSL-ASL gloss pairs. No deep learning model is trained and no new feature is extracted.

## Dataset Summary
- video rows after merge: 16342
- unique videos: 16342
- gloss rows: 5314
- numeric feature columns used before gloss aggregation: 131
- mapped VSL-ASL pairs scored: 392

## Similarity Statistics
- Mean Similarity: 0.557183
- Median: 0.553370
- Standard Deviation: 0.170698
- Minimum: 0.147703
- Maximum: 0.928333

## Top Similar
- cửa sổ ↔ window: 0.9283
- i ↔ i: 0.9151
- mùa đông ↔ winter: 0.8994
- bụng ↔ abdomen: 0.8949
- ông ngoại ↔ grandfather: 0.8940
- ai ↔ who: 0.8913
- a ↔ a: 0.8899
- bên dưới ↔ under: 0.8786
- Trung Quốc (nước Trung Quốc) ↔ china: 0.8771
- dao ↔ knife: 0.8759

## Top Different
- xe tải ↔ truck: 0.1477; main differences: movement (50.9%); handshape (43.9%); orientation (5.2%)
- ngày sinh ↔ birthday: 0.1997; main differences: movement (53.0%); handshape (43.1%); orientation (3.9%)
- phòng thư viện ↔ library: 0.2009; main differences: movement (49.7%); handshape (44.7%); orientation (5.6%)
- quan trọng ↔ important: 0.2161; main differences: movement (54.8%); handshape (39.4%); orientation (5.5%)
- Mexico (nước Mexico) ↔ mexico: 0.2165; main differences: movement (80.8%); handshape (16.8%); orientation (2.1%)
- con chó ↔ dog: 0.2191; main differences: movement (71.3%); handshape (25.3%); orientation (3.4%)
- con muỗi ↔ mosquito: 0.2202; main differences: handshape (55.5%); movement (37.9%); orientation (6.3%)
- bức tranh ↔ picture: 0.2244; main differences: handshape (62.1%); movement (32.2%); orientation (5.6%)
- chọn lựa ↔ choose: 0.2249; main differences: handshape (50.9%); movement (42.4%); orientation (6.3%)
- giận dữ ↔ angry: 0.2502; main differences: movement (74.3%); handshape (19.8%); orientation (5.1%)

## Regional Similarity
- No VSL gloss with `_B`, `_T`, `_N` regional suffix was detected with enough paired regions.

## Automatic Notes
- Average VSL-ASL feature similarity is moderate; top and bottom glosses should be inspected separately.
- Similarity scores are widely dispersed, so individual gloss-level analysis is important.

## Warnings
- Mapping file used exactly: E:\NAM3\BDML\Project\DBML\compare\src\src4_8\mapping.csv with columns `vsl_label` and `asl_label`.
