import os

import pandas as pd

from utils_4_6 import OUTPUT_DIR, read_csv, write_csv


def top_component(summary, group_cols, value_col="percentage", top_n=5):
    rows = []
    for dataset, part in summary.groupby("dataset"):
        part = part.sort_values(value_col, ascending=False).head(top_n)
        for _, row in part.iterrows():
            label = " / ".join(str(row[col]) for col in group_cols)
            rows.append(
                {
                    "dataset": dataset,
                    "component": label,
                    "count": row["count"],
                    "percentage": row[value_col],
                }
            )
    return pd.DataFrame(rows)


def main():
    handshape = read_csv("handshape_cluster_summary.csv")
    location = read_csv("location_statistics.csv")
    orientation = read_csv("orientation_statistics.csv")
    one_two = read_csv("one_two_hand_statistics.csv")

    handshape_top5 = top_component(
        handshape,
        ["handshape_cluster", "cluster_label_proxy"],
        top_n=5,
    )
    handshape_top5["analysis"] = "handshape_top5"

    location_top = top_component(location, ["dominant_location"], top_n=5)
    location_top["analysis"] = "location"

    orientation_top = top_component(
        orientation,
        ["dominant_finger_direction", "dominant_palm_orientation"],
        top_n=5,
    )
    orientation_top["analysis"] = "orientation"

    one_two_top = top_component(one_two, ["activity_type"], top_n=5)
    one_two_top["analysis"] = "one_two_hand"

    comparison = pd.concat(
        [handshape_top5, location_top, orientation_top, one_two_top],
        ignore_index=True,
    )
    write_csv(comparison, "dataset_summary.csv")

    report_lines = [
        "# 4.6 Structural Analysis Summary",
        "",
        "Note: this analysis intentionally focuses on hands and pose to match the structural-analysis goal.",
        "Face keypoints are not used even though the newer extraction has much lower face missing rates.",
        "",
    ]
    for analysis, part in comparison.groupby("analysis"):
        report_lines.append(f"## {analysis}")
        for dataset, ds_part in part.groupby("dataset"):
            total = ds_part["percentage"].sum()
            report_lines.append(f"- {dataset}: top components cover {total:.2f}%")
            for _, row in ds_part.iterrows():
                report_lines.append(
                    f"  - {row['component']}: {row['percentage']:.2f}% ({int(row['count'])})"
                )
        report_lines.append("")

    path = os.path.join(OUTPUT_DIR, "dataset_summary_report.md")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines))

    print("Saved dataset_summary.csv and dataset_summary_report.md")


if __name__ == "__main__":
    main()
