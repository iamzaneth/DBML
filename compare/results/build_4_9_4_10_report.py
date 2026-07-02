from __future__ import annotations

import csv
import html
import math
import zipfile
from pathlib import Path


COMPARE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = COMPARE_ROOT / "results"
OUT_49 = RESULTS / "output_4_9"
OUT_410 = RESULTS / "output_4_10"
DOCX_PATH = RESULTS / "bao_cao_phan_tich_output_4_9_4_10.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str | None) -> float:
    try:
        return float(value or "nan")
    except ValueError:
        return math.nan


def fmt(value: str | float, digits: int = 3) -> str:
    number = to_float(str(value))
    if math.isnan(number):
        return ""
    if abs(number) >= 100:
        return f"{number:.1f}"
    return f"{number:.{digits}f}"


def top_rows(rows: list[dict[str, str]], dataset: str, metric: str, n: int = 5) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("dataset") == dataset]
    return sorted(selected, key=lambda row: to_float(row.get(metric)), reverse=True)[:n]


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def paragraph(text: str = "", style: str | None = None, bold: bool = False, italic: bool = False) -> str:
    p_style = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    b = "<w:b/>" if bold else ""
    i = "<w:i/>" if italic else ""
    return (
        f"<w:p>{p_style}<w:r><w:rPr>{b}{i}</w:rPr>"
        f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>'
    )


def heading(text: str, level: int = 1) -> str:
    return paragraph(text, f"Heading{level}")


def bullet(text: str) -> str:
    return paragraph("• " + text)


def table(headers: list[str], rows: list[list[object]]) -> str:
    tbl = [
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="0" w:type="auto"/></w:tblPr>'
    ]
    all_rows = [headers] + rows
    for ridx, row in enumerate(all_rows):
        tbl.append("<w:tr>")
        for cell in row:
            bold = "<w:b/>" if ridx == 0 else ""
            tbl.append(
                "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
                f"<w:p><w:r><w:rPr>{bold}</w:rPr><w:t>{esc(cell)}</w:t></w:r></w:p></w:tc>"
            )
        tbl.append("</w:tr>")
    tbl.append("</w:tbl>")
    return "".join(tbl)


def image_paragraph(rel_id: str, caption: str, cx: int = 5486400, cy: int = 3657600) -> str:
    drawing = f"""
    <w:p><w:r><w:drawing><wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" distT="0" distB="0" distL="0" distR="0">
      <wp:extent cx="{cx}" cy="{cy}"/>
      <wp:docPr id="{rel_id[3:]}" name="{esc(caption)}"/>
      <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
        <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
          <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:nvPicPr><pic:cNvPr id="0" name="{esc(caption)}"/><pic:cNvPicPr/></pic:nvPicPr>
            <pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
            <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
          </pic:pic>
        </a:graphicData>
      </a:graphic>
    </wp:inline></w:drawing></w:r></w:p>
    """
    return drawing + paragraph(caption, italic=True)


def media_name(idx: int, image: Path) -> str:
    return f"image_{idx:02d}_{image.parent.name}_{image.name}"


def docx_parts(body: str, images: list[Path]) -> dict[str, bytes]:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    ]
    content_types = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Default Extension="png" ContentType="image/png"/>',
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>',
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>',
    ]
    for idx, image in enumerate(images, start=2):
        rels.append(
            f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{media_name(idx, image)}"/>'
        )
    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}"><w:body>{body}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr></w:body></w:document>"""
    styles = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="30"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr></w:style>
</w:styles>"""
    return {
        "[Content_Types].xml": f'<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">{"".join(content_types)}</Types>'.encode("utf-8"),
        "_rels/.rels": b'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>',
        "word/document.xml": document.encode("utf-8"),
        "word/styles.xml": styles.encode("utf-8"),
        "word/_rels/document.xml.rels": f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{"".join(rels)}</Relationships>'.encode("utf-8"),
    }


def main() -> None:
    difficulty = read_csv(OUT_49 / "difficulty_summary.csv")
    signer = read_csv(OUT_49 / "signer_variation.csv")
    motion = read_csv(OUT_49 / "motion_complexity_ranking.csv")
    sequence = read_csv(OUT_49 / "sequence_length_variation.csv")
    comparison = read_csv(OUT_410 / "comparison_summary.csv")
    vsl_nearest = read_csv(OUT_410 / "VSL" / "top20_nearest_gloss_pairs_euclidean.csv")
    asl_nearest = read_csv(OUT_410 / "ASL" / "top20_nearest_gloss_pairs_euclidean.csv")
    vsl_disp = read_csv(OUT_410 / "VSL" / "top20_dispersed_glosses.csv")
    asl_disp = read_csv(OUT_410 / "ASL" / "top20_dispersed_glosses.csv")

    images = [
        OUT_49 / "figures" / "signer_variation_top20_VSL.png",
        OUT_49 / "figures" / "signer_variation_top20_ASL.png",
        OUT_49 / "figures" / "motion_complexity_top20_VSL.png",
        OUT_49 / "figures" / "motion_complexity_top20_ASL.png",
        OUT_49 / "figures" / "sequence_length_variation_top20_VSL.png",
        OUT_49 / "figures" / "sequence_length_variation_top20_ASL.png",
        OUT_410 / "VSL" / "pca.png",
        OUT_410 / "ASL" / "pca.png",
        OUT_410 / "VSL" / "tsne.png",
        OUT_410 / "ASL" / "tsne.png",
        OUT_410 / "VSL" / "umap.png",
        OUT_410 / "ASL" / "umap.png",
    ]
    images = [image for image in images if image.exists()]
    rel_by_image = {image: f"rId{idx}" for idx, image in enumerate(images, start=2)}

    body: list[str] = []
    body.append(paragraph("Phân tích output 4.9 và 4.10", "Title"))
    body.append(paragraph("Báo cáo này tổng hợp và diễn giải các kết quả đầu ra của phần 4.9 Dataset-level Recognition Difficulty Analysis và phần 4.10 Feature Space Analysis. Nội dung bám theo khung phân tích đã đề xuất, sử dụng trực tiếp các bảng CSV và hình trong thư mục kết quả."))

    body.append(heading("4.9. Dataset-level Recognition Difficulty Analysis", 1))
    body.append(heading("4.9.1. Mục tiêu", 2))
    body.append(paragraph("Mục tiêu của phần 4.9 là đánh giá độ khó nhận diện của từng gloss dựa trên đặc điểm nội tại của dữ liệu. Khác với các phần 4.6, 4.7 và 4.8, phần này không tập trung mô tả riêng từng loại đặc trưng hoặc so sánh giữa VSL và ASL, mà tổng hợp các dấu hiệu gây khó cho mô hình nhận diện trong từng dataset. Ba nguyên nhân chính được xem xét gồm signer variation, motion complexity và sequence length variation."))
    body.append(heading("4.9.2. Input và phương pháp", 2))
    body.append(paragraph("Phân tích không trích xuất thêm đặc trưng mới. Các vector đầu vào được ghép từ handshape, location, orientation ở phần 4.6 và motion features ở phần 4.7. Với signer variation, mỗi video được biểu diễn bằng một feature vector tổng hợp; centroid của từng gloss được tính trong không gian đặc trưng, sau đó khoảng cách từ từng video đến centroid được dùng để đo độ biến thiên nội lớp. Với motion complexity và sequence length variation, báo cáo sử dụng trực tiếp các thống kê theo gloss từ output 4.7."))
    body.append(table(
        ["Dataset", "Analysis", "Số gloss", "Mean", "Median", "Min", "Max"],
        [[r["dataset"], r["analysis"], r["num_glosses"], fmt(r["mean"]), fmt(r["median"]), fmt(r["min"]), fmt(r["max"])] for r in difficulty],
    ))

    body.append(heading("4.9.3. Signer Variation Analysis", 2))
    body.append(paragraph("Signer variation phản ánh mức độ khác nhau giữa các lần thực hiện cùng một gloss. Mean distance càng lớn cho thấy các video cùng nhãn càng phân tán quanh centroid, từ đó mô hình khó học một mẫu biểu diễn ổn định cho gloss đó."))
    body.append(paragraph("Ở VSL, mean distance trung bình là 0.668 và median bằng 0.000. Điều này cho thấy nhiều gloss có rất ít mẫu hoặc các mẫu sau khi chuẩn hóa khá gần nhau, nhưng vẫn tồn tại một nhóm gloss có biến thiên rất mạnh. Gloss e thẹn có mean distance 32.171, cao nhất toàn bộ VSL, tiếp theo là màn và quần bò. Các gloss này nên được xem là nhóm khó vì cùng một nhãn nhưng biểu hiện không gian đặc trưng không ổn định."))
    body.append(paragraph("Ở ASL, mean distance trung bình là 5.039 và median là 5.270, cao hơn rõ trong chính phân bố ASL. Các gloss dance, underwear, after và nervous có signer variation lớn, cho thấy người ký có thể thực hiện cùng một gloss với khác biệt đáng kể về hình dạng tay, vị trí, hướng tay hoặc quỹ đạo chuyển động."))
    for dataset in ("VSL", "ASL"):
        body.append(table(
            [f"Top {dataset}", "Gloss", "Samples", "Mean distance", "Within-class variance"],
            [[row["dataset"], row["gloss"], row["num_samples"], fmt(row["mean_distance"]), fmt(row["within_class_variance"])] for row in top_rows(signer, dataset, "mean_distance")],
        ))
    for image in images[:2]:
        body.append(image_paragraph(rel_by_image[image], f"Hình: Top 20 signer variation - {image.stem.split('_')[-1]}"))

    body.append(heading("4.9.4. Motion Complexity Analysis", 2))
    body.append(paragraph("Motion complexity đo mức độ phức tạp của chuyển động trong từng gloss. Chỉ số này tăng khi tổng chuyển động, vận tốc, gia tốc, độ biến thiên chuyển động hoặc số lần đổi hướng lớn. Motion càng phức tạp thì temporal pattern càng khó học, đặc biệt với các mô hình cần căn chỉnh chuỗi theo thời gian."))
    body.append(paragraph("Trong VSL, motion complexity trung bình là 0.090, nhưng gloss đuôi đạt 0.806, nổi bật do total motion 132.542, mean velocity 92.379 và 28 lần đổi hướng. Một số gloss như gà mái, cổ tích, tóc dựng đứng cũng có điểm cao, chủ yếu do quỹ đạo dài hoặc số lần đổi hướng lớn. Tuy nhiên, nhiều gloss VSL trong nhóm đầu chỉ có một mẫu, nên các kết luận ở mức gloss cần được đọc như dấu hiệu rủi ro dữ liệu hơn là đặc tính ổn định tuyệt đối của ký hiệu."))
    body.append(paragraph("Trong ASL, motion complexity trung bình là 0.198. Các gloss ceiling, shampoo, climb và swimming đứng đầu, đều có điểm trên 0.64. Nhóm này có chuyển động lặp, đổi hướng nhiều hoặc biên độ tay rõ, vì vậy mô hình cần học tốt cả hướng chuyển động và nhịp chuyển động để phân biệt."))
    for dataset in ("VSL", "ASL"):
        body.append(table(
            [f"Top {dataset}", "Gloss", "Samples", "Complexity", "Total motion", "Velocity", "Direction changes"],
            [[row["dataset"], row["gloss"], row["num_samples"], fmt(row["motion_complexity_score"]), fmt(row["mean_total_motion"]), fmt(row["mean_velocity"]), fmt(row["direction_change_mean"])] for row in top_rows(motion, dataset, "motion_complexity_score")],
        ))
    for image in images[2:4]:
        body.append(image_paragraph(rel_by_image[image], f"Hình: Top 20 motion complexity - {image.stem.split('_')[-1]}"))

    body.append(heading("4.9.5. Sequence Length Variation Analysis", 2))
    body.append(paragraph("Sequence length variation đánh giá mức độ dao động số frame giữa các video cùng gloss. Variance càng lớn thì việc căn chỉnh thời gian càng khó, vì cùng một nhãn có thể được thực hiện nhanh, chậm hoặc kéo dài khác nhau. Đây là nguồn nhiễu quan trọng đối với các mô hình chuỗi."))
    body.append(paragraph("Trong VSL, variance trung bình là 57.120 nhưng max đạt 7200.000 ở gloss bạn. Các gloss quả đu đủ, tay chân sạch sẽ, ở ngoài và vật nuôi cũng có variance rất cao. Điều này cho thấy một số gloss có độ dài thực hiện không nhất quán, làm tăng khó khăn cho bước temporal modeling."))
    body.append(paragraph("Trong ASL, variance trung bình là 95.639 và median là 50.229. Gloss april có variance 2666.400, tiếp theo là plant, article, bank và march. Nhóm này cần được chú ý khi huấn luyện vì mô hình có thể học sai nếu padding, sampling hoặc chuẩn hóa độ dài chuỗi không đủ ổn định."))
    for dataset in ("VSL", "ASL"):
        body.append(table(
            [f"Top {dataset}", "Gloss", "Samples", "Mean frames", "Std frames", "Variance"],
            [[row["dataset"], row["gloss"], row["num_samples"], fmt(row["mean_frames"]), fmt(row["std_frames"]), fmt(row["sequence_length_variance"])] for row in top_rows(sequence, dataset, "sequence_length_variance")],
        ))
    for image in images[4:6]:
        body.append(image_paragraph(rel_by_image[image], f"Hình: Top 20 sequence length variation - {image.stem.split('_')[-1]}"))

    body.append(heading("4.9.6. Kết luận phần 4.9", 2))
    body.append(paragraph("Kết quả 4.9 cho thấy độ khó nhận diện không đến từ một nguồn duy nhất. Một gloss có thể khó vì người ký thực hiện không nhất quán, vì chuyển động phức tạp, hoặc vì độ dài chuỗi biến thiên mạnh. Do đó, các gloss đứng đầu trong ba bảng ranking nên được ưu tiên kiểm tra dữ liệu, cân bằng số mẫu, chuẩn hóa temporal length và đánh giá lỗi mô hình theo từng gloss."))

    body.append(heading("4.10. Feature Space Analysis", 1))
    body.append(heading("4.10.1. Mục tiêu và input", 2))
    body.append(paragraph("Mục tiêu của phần 4.10 là đánh giá chất lượng không gian đặc trưng được xây dựng từ handshape, location, orientation và motion. Mỗi video được biểu diễn bằng một vector 80 chiều sau khi ghép đặc trưng và chuẩn hóa. Phân tích trả lời ba câu hỏi: feature có đủ khả năng phân biệt gloss không, các lớp có tách nhau tốt không, và cluster có bị chồng lấn không."))
    body.append(table(
        ["Dataset", "Samples", "Glosses", "Feature dim", "PCA PC1", "PCA PC1+PC2", "Silhouette", "DBI", "Mean compactness", "Mean separation"],
        [[r["dataset"], r["num_samples"], r["num_glosses"], r["feature_dimension"], fmt(r["PCA explained variance PC1"]), fmt(r["PCA explained variance cumulative PC1_PC2"]), fmt(r["Silhouette Score"]), fmt(r["Davies-Bouldin Index"]), fmt(r["Mean Compactness"]), fmt(r["Mean Euclidean Separation"])] for r in comparison],
    ))

    body.append(heading("4.10.2. PCA", 2))
    body.append(paragraph("PCA cho thấy hai thành phần đầu chỉ giải thích 39.596% phương sai ở VSL và 34.484% ở ASL. Điều này hàm ý cấu trúc phân biệt gloss không thể được biểu diễn đầy đủ trên mặt phẳng 2D tuyến tính. Nếu quan sát PCA plot, các điểm có xu hướng tạo nhiều vùng gần nhau thay vì các cụm tách rời hoàn toàn. PCA vì vậy hữu ích để phát hiện outlier và xu hướng phân bố tổng quát, nhưng chưa đủ để khẳng định feature space đã phân lớp tốt."))
    for image in images[6:8]:
        body.append(image_paragraph(rel_by_image[image], f"Hình: PCA feature space - {image.parent.name}"))

    body.append(heading("4.10.3. t-SNE và UMAP", 2))
    body.append(paragraph("t-SNE được dùng để quan sát cấu trúc cục bộ và local neighborhood. Nếu các video cùng gloss nằm gần nhau, feature đang giữ được quan hệ gần trong lớp; ngược lại, các vùng trộn màu biểu thị overlap. UMAP bổ sung góc nhìn toàn cục hơn, giúp xem các nhóm gloss có tạo thành các manifold riêng hay vẫn chen lẫn."))
    body.append(paragraph("Kết quả định lượng đi kèm cho thấy silhouette âm ở cả VSL (-0.091) và ASL (-0.237), vì vậy các đồ thị t-SNE/UMAP nên được diễn giải theo hướng feature space có một số cụm cục bộ nhưng chưa tách lớp rõ ràng trên toàn bộ dataset. ASL có silhouette thấp hơn và Davies-Bouldin Index cao hơn, cho thấy overlap nội tại trong không gian đặc trưng mạnh hơn."))
    for image in images[8:12]:
        body.append(image_paragraph(rel_by_image[image], f"Hình: {image.stem.upper()} feature space - {image.parent.name}"))

    body.append(heading("4.10.4. Intra-class Compactness", 2))
    body.append(paragraph("Intra-class compactness là khoảng cách trung bình từ các video của một gloss đến centroid của gloss đó. VSL có mean compactness 0.635 và median 0.000, phản ánh nhiều gloss có rất ít mẫu hoặc cụm rất chặt. Tuy nhiên, các gloss như e thẹn, màn, chuyên gia và quần bò vẫn có compactness cao, cho thấy nguy cơ phân tán nội lớp. ASL có mean compactness 5.313 và median 5.597; các gloss underwear, monster, russia, yes và bike là các cụm phân tán mạnh."))
    body.append(table(
        ["Dataset", "Gloss", "Samples", "Compactness", "Median dist", "Max dist"],
        [["VSL", r["gloss"], r["num_samples"], fmt(r["compactness"]), fmt(r["median_distance"]), fmt(r["max_distance"])] for r in vsl_disp[:5]]
        + [["ASL", r["gloss"], r["num_samples"], fmt(r["compactness"]), fmt(r["median_distance"]), fmt(r["max_distance"])] for r in asl_disp[:5]],
    ))

    body.append(heading("4.10.5. Inter-class Separation", 2))
    body.append(paragraph("Inter-class separation được tính bằng khoảng cách giữa centroid của các gloss. Khoảng cách càng nhỏ nghĩa là hai gloss càng dễ bị nhầm trong feature space. Ở VSL, cặp gần nhất là héc-tô-gam (hg) và héc-tô-mét vuông (hm2) với Euclidean distance 0.870. Ở ASL, cặp private và secret có khoảng cách 0.414, là dấu hiệu overlap rất mạnh giữa hai centroid. Nhiều cặp gần nhất cũng có quan hệ ngữ nghĩa hoặc hình thái gần nhau, ví dụ already/done, describe/explain, freeway/highway."))
    body.append(table(
        ["Dataset", "Gloss A", "Gloss B", "Euclidean", "Cosine"],
        [["VSL", r["gloss_a"], r["gloss_b"], fmt(r["euclidean_distance"]), fmt(r["cosine_distance"], 4)] for r in vsl_nearest[:5]]
        + [["ASL", r["gloss_a"], r["gloss_b"], fmt(r["euclidean_distance"]), fmt(r["cosine_distance"], 4)] for r in asl_nearest[:5]],
    ))

    body.append(heading("4.10.6. Silhouette Score và Davies-Bouldin Index", 2))
    body.append(paragraph("Silhouette Score đồng thời đo độ gắn kết trong lớp và độ tách biệt giữa các lớp. Giá trị gần 1 là tốt, gần 0 là overlap, và nhỏ hơn 0 cho thấy nhiều mẫu gần cụm khác hơn cụm của chính nó. Kết quả VSL = -0.091 và ASL = -0.237 cho thấy feature space hiện tại chưa đủ tách biệt cho toàn bộ gloss. Davies-Bouldin Index càng nhỏ càng tốt; VSL đạt 1.424, trong khi ASL đạt 3.717, củng cố nhận định rằng ASL có mức overlap cluster cao hơn trong output này."))
    body.append(paragraph("Nhìn chung, các feature handshape, location, orientation và motion đã tạo ra một không gian có tín hiệu phân biệt nhất định, thể hiện qua khoảng cách centroid trung bình khá lớn. Tuy nhiên, compactness không đồng đều, silhouette âm và các cặp centroid rất gần cho thấy feature space vẫn còn chồng lấn. Vì vậy, để cải thiện nhận diện gloss, nên kết hợp thêm đặc trưng động theo chuỗi, attention theo vùng tay-mặt-thân, hoặc học biểu diễn bằng mô hình temporal/deep embedding thay vì chỉ dựa trên thống kê video-level."))

    body.append(heading("Kết luận chung", 1))
    body.append(paragraph("Phần 4.9 chỉ ra các gloss khó ở cấp dataset thông qua ba nguồn: signer variation, motion complexity và sequence length variation. Phần 4.10 cho thấy không gian đặc trưng tổng hợp có khả năng mô tả dữ liệu nhưng chưa tách lớp mạnh, đặc biệt khi đánh giá bằng silhouette và Davies-Bouldin. Hai phần bổ trợ cho nhau: 4.9 giúp xác định gloss nào có nguy cơ gây lỗi, còn 4.10 giải thích mức độ overlap của feature space làm nền cho bài toán phân loại."))

    parts = docx_parts("".join(body), images)
    with zipfile.ZipFile(DOCX_PATH, "w", zipfile.ZIP_DEFLATED) as docx:
        for name, data in parts.items():
            docx.writestr(name, data)
        for idx, image in enumerate(images, start=2):
            docx.write(image, f"word/media/{media_name(idx, image)}")
    print(DOCX_PATH)


if __name__ == "__main__":
    main()
