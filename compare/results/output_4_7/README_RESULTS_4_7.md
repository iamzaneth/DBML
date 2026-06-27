# Section 4.7 - Motion Feature Analysis Results
## Files generated
- `motion_features_per_video.csv`: one row per video/sample.
- `motion_summary_by_dataset.csv`: dataset-level long summary.
- `motion_summary_by_dataset_wide.csv`: dataset-level mean comparison table.
- `sequence_length_by_label.csv`: sequence length statistics per label.
- `motion_features_by_label.csv`: label-level motion statistics.
- `top_complex_motion_labels.csv`: labels with highest motion complexity score.
- `dtw/dtw_intra_class_summary.csv`: intra-class trajectory variability.
- `dtw/dtw_inter_class_candidate_pairs.csv`: likely confusing inter-class trajectory pairs.
- `figures/`: plots for comparison and reporting.

## How to interpret key metrics
- `total_motion`: total hand displacement; higher means larger movement amplitude.
- `mean_velocity`: average per-frame movement speed; higher means faster signing.
- `motion_variance`: movement stability; higher means more irregular motion.
- `sequence_length_variance`: temporal consistency within a label; higher means samples of the same sign have different durations.
- `trajectory_length`: total path length of both hands.
- `straightness_ratio`: close to 1 = straight path; lower = curved/repeated/complex path.
- `left_right_hand_dist_mean`: average distance between two hands; useful for two-hand interaction analysis.
- `intra_class_dtw_mean`: same-label movement variability; higher means signers perform the same label differently.
- `inter_class dtw_distance`: lower means different labels have similar trajectories and may be confused.

## Quick dataset comparison
- `total_motion` is higher on average in **VSL**.
- `mean_velocity` is higher on average in **VSL**.
- `motion_variance` is higher on average in **VSL**.
- `trajectory_length` is higher on average in **VSL**.
- `straightness_ratio` is higher on average in **ASL**.
- `num_frames` is higher on average in **VSL**.

## Suggested conclusion template
> Based on the motion feature analysis, the dataset with higher total motion, velocity, motion variance, and intra-class DTW can be considered more temporally complex. High sequence length variance indicates inconsistent sign duration within the same label, while low inter-class DTW reveals pairs of signs with similar movement trajectories that may cause recognition confusion.
