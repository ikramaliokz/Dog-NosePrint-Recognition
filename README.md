# Ensemble Prediction Process

# Overview

We employed the `weighted_boxes_fusion` function from the `ensemble_boxes` library, an open-source tool, to effectively combine predictions from three distinct segmentation models. This method enhances accuracy by merging the strengths of each model into a unified prediction, leveraging the robust methodology proposed in the referenced paper (https://arxiv.org/abs/1910.13302). For further details and implementation specifics, the GitHub repository of the library can be accessed at https://github.com/ZFTurbo/Weighted-Boxes-Fusion.


## Models
- **YOLOv8**
- **YOLACT**
- **OneFormer**

## Code Workflow
1. **Collect Outputs**: Gather bounding boxes, scores, and class labels from each model:
    - `boxes_list = [yolo_boxes, yola_boxes, list(oneformer_boxes)]`
    - `scores_list = [yolo_scores, yola_scores, oneformer_score]`
    - `labels_list = [[0], [0], [0]]` (Assuming single class detection)

2. **Set Parameters**:
    - Equal weights: `weights = [1, 1, 1]`
    - IoU threshold: `iou_thr = 0.5`
    - Skip box threshold: `skip_box_thr = 0.0001`

3. **Apply Fusion**:
    - Use `weighted_boxes_fusion` to combine predictions:
    ```python
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )
    ```

## Conclusion
The fusion method enhances prediction accuracy by combining the strengths of the individual models and reducing false positives.
