# Report 4.8 - Similarity Analysis between VSL and WLASL

## Pipeline
The pipeline reads available CSV outputs, validates merge keys, merges per-video feature tables, aggregates video features into gloss-level vectors, standardizes all gloss vectors with one StandardScaler, then computes cosine similarity for mapped VSL-ASL gloss pairs. No deep learning model is trained and no new feature is extracted.

## Dataset Summary
- video rows after merge: 16342
- unique videos: 16342
- gloss rows: 5314
- numeric feature columns used before gloss aggregation: 80
- mapped VSL-ASL pairs scored: 392

## Similarity Statistics
- Mean Similarity: 0.570063
- Median: 0.555423
- Standard Deviation: 0.176582
- Minimum: 0.150502
- Maximum: 0.974185

## Top Similar
- bụng ↔ abdomen: 0.9742
- i ↔ i: 0.9674
- a ↔ a: 0.9634
- cây nến ↔ candle: 0.9536
- cửa sổ ↔ window: 0.9530
- bố ↔ dad: 0.9285
- mùa đông ↔ winter: 0.9211
- bên dưới ↔ under: 0.9172
- bên dưới ↔ below: 0.9157
- xe máy ↔ motorcycle: 0.9150

## Top Different
- phòng thư viện ↔ library: 0.1505; main differences: handshape (72.6%); movement (17.9%); orientation (9.5%)
- Mexico (nước Mexico) ↔ mexico: 0.1821; main differences: movement (64.5%); handshape (30.1%); orientation (4.6%)
- con muỗi ↔ mosquito: 0.2019; main differences: handshape (78.5%); movement (11.5%); orientation (9.7%)
- bức tranh ↔ picture: 0.2146; main differences: handshape (77.3%); movement (15.3%); orientation (7.4%)
- ngày sinh ↔ birthday: 0.2150; main differences: handshape (64.0%); movement (29.8%); orientation (6.1%)
- xe tải ↔ truck: 0.2174; main differences: handshape (64.3%); movement (27.0%); orientation (8.7%)
- giá sách ↔ shelf: 0.2234; main differences: handshape (80.3%); movement (10.9%); orientation (8.9%)
- quan trọng ↔ important: 0.2300; main differences: handshape (68.6%); movement (21.8%); orientation (9.0%)
- bạn ↔ friend: 0.2339; main differences: handshape (76.0%); movement (17.7%); orientation (6.3%)
- bánh mì ↔ bread: 0.2341; main differences: handshape (78.8%); movement (16.3%); orientation (4.9%)

## Regional Similarity
- No VSL gloss with `_B`, `_T`, `_N` regional suffix was detected with enough paired regions.

## Automatic Notes
- Average VSL-ASL feature similarity is moderate; top and bottom glosses should be inspected separately.
- Similarity scores are widely dispersed, so individual gloss-level analysis is important.

## Warnings
- Mapping file used exactly: D:\Project\DBML\compare\src\src4_8\mapping.csv with columns `vsl_label` and `asl_label`.
