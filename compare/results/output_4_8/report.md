# Report 4.8 - Similarity Analysis between VSL and WLASL

## Pipeline
The pipeline reads available CSV outputs, validates merge keys, merges per-video feature tables, aggregates video features into gloss-level vectors, standardizes all gloss vectors with one StandardScaler, then computes cosine similarity for mapped VSL-ASL gloss pairs. No deep learning model is trained and no new feature is extracted.

## Dataset Summary
- video rows after merge: 16342
- unique videos: 16342
- gloss rows: 5314
- schema-selected feature columns used before gloss aggregation: 82
- mapped VSL-ASL pairs scored: 392

## Similarity Statistics
- Mean Similarity: 0.569270
- Median: 0.555481
- Standard Deviation: 0.174203
- Minimum: 0.145227
- Maximum: 0.974185

## Top Similar
- bụng ↔ abdomen: 0.9742
- i ↔ i: 0.9674
- a ↔ a: 0.9634
- cửa sổ ↔ window: 0.9485
- cây nến ↔ candle: 0.9460
- bố ↔ dad: 0.9285
- mùa đông ↔ winter: 0.9213
- bên dưới ↔ under: 0.9179
- xe máy ↔ motorcycle: 0.9149
- 6 ↔ six: 0.9142

## Top Different
- phòng thư viện ↔ library: 0.1452; main differences: handshape (69.6%); movement (17.1%); orientation (11.2%)
- Mexico (nước Mexico) ↔ mexico: 0.1829; main differences: movement (63.6%); handshape (29.6%); orientation (5.3%)
- con muỗi ↔ mosquito: 0.1983; main differences: handshape (75.8%); movement (11.1%); orientation (11.0%)
- bức tranh ↔ picture: 0.2227; main differences: handshape (76.4%); movement (15.1%); orientation (7.9%)
- giá sách ↔ shelf: 0.2240; main differences: handshape (80.0%); movement (10.8%); orientation (9.0%)
- ngày sinh ↔ birthday: 0.2266; main differences: handshape (62.6%); movement (29.1%); orientation (7.0%)
- quan trọng ↔ important: 0.2271; main differences: handshape (66.2%); movement (21.0%); orientation (10.5%)
- bánh mì ↔ bread: 0.2398; main differences: handshape (78.6%); movement (16.2%); orientation (5.0%)
- bàn ghế ↔ table: 0.2466; main differences: handshape (80.9%); orientation (9.9%); movement (8.8%)
- địa chỉ ↔ address: 0.2486; main differences: handshape (75.8%); movement (13.6%); orientation (9.4%)

## Regional Similarity
- No VSL gloss with `_B`, `_T`, `_N` regional suffix was detected with enough paired regions.

## Automatic Notes
- Average VSL-ASL feature similarity is moderate; top and bottom glosses should be inspected separately.
- Similarity scores are widely dispersed, so individual gloss-level analysis is important.

## Warnings
- Mapping file used exactly: E:\NAM3\BDML\Project\DBML\compare\src\src4_8\mapping.csv with columns `vsl_label` and `asl_label`.
