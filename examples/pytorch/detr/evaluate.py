import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import AutoImageProcessor
from typing import Optional, Mapping, Tuple
from transformers.image_transforms import center_to_corners_format
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

#inference related variables and post-processing functions
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]    

def plot_results(pil_img, scores, labels, boxes, tag="", id2label=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        #text = f'{model.config.id2label[label]}: {score:0.2f}'
        text = f'{id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig('output_image' + tag + '.jpg')

### Reference:
### https://huggingface.co/docs/transformers/en/tasks/object_detection#preparing-function-to-compute-map
### Derived source from:
### https://github.com/huggingface/transformers/blob/main/examples/pytorch/object-detection/run_object_detection.py


def convert_bbox_yolo_to_pascal(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes

def post_process_model_outputs(
    predictions, # List of pre-processed predictions. Each entry is model output for one batch of input images submitted to model.
    targets, #List of list of pre-processed targets. Each Inner list corresponds to a batch submitted to model
    image_processor: AutoImageProcessor,
    threshold: float = 0.0,
    id2label: Optional[Mapping[int, str]] = None,
    image_batch_list = None #List of List of PIL Image Batches to annotate results with. Each Inner list corresponds to a batch submitted to model
) -> Mapping[str, float]:
    """
    Generate image representing the detections from the model

    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"
    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        batch_image_sizes = torch.tensor([x["orig_size"].numpy() for x in batch])
        image_sizes.append(batch_image_sizes)

        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            tboxes = torch.tensor(image_target["boxes"])
            tboxes = convert_bbox_yolo_to_pascal(tboxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            tboxes = tboxes.to('cpu')
            labels = labels.to('cpu')
            post_processed_targets.append({"boxes": tboxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    # Along the way, save annotated output images
    print(f'image_sizes = {image_sizes}')
    count = 0
    for batch, target_sizes, image_batch in zip(predictions, image_sizes, image_batch_list):
        post_processed_output = image_processor.post_process_object_detection(
                                batch, threshold=threshold, target_sizes=target_sizes
                            )
    
        # Draw image with boxes and save it as a file
        try:
            for results, image in zip(post_processed_output, image_batch):
                plot_results(image, results['scores'], results['labels'], results['boxes'], id2label=id2label, tag=str(count))
                count += 1
        except Exception as e:
            print(f'Exception in saving annotated image at count={count}. Skipping')
            print(f'Error message: {str(e)}')
    
        # TODO: We seem to encounter a mix of HPU and CPU tensors when below sequence is invoked
        #       without converting above tensors to CPU. Need to root-cause and fix that later.
        # For now, we simply WA that problem by moving all tensors in post_processed_* to CPU 
        for dict_item in post_processed_output:
            for key in dict_item:
                dict_item[key] = dict_item[key].to('cpu')
        post_processed_predictions.extend(post_processed_output)
    
    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}
    return metrics
