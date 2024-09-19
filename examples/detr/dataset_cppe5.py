import os
import torch
import torchvision
import numpy as np
from arguments import args
from transformers import DetrImageProcessor
from torch.utils.data import DataLoader

# Init variables for this script.
num_labels = 5  #TODO: automate this from json files in dataset instead of hardcoding
buckets = None
train_dataloader = None
val_dataloader = None

img_folder = '/root/gs-274/CPPE-Dataset/data/images'
annotations_folder = '/root/gs-274/CPPE-Dataset/data/annotations'
train_ann_file = os.path.join(annotations_folder, "train.json")
val_ann_file = os.path.join(annotations_folder, "test.json")

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, processor, train=True):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension
        return pixel_values, target

def get_num_labels():
    # Should feed max-label ID plus 1 to the Detr class.
    # For an explanation see below
    # https://github.com/facebookresearch/detr/issues/108
    return num_labels + 1

def bucketer(dl, num_buckets):
    shapes = []
    for idx, dt in enumerate(dl):
        shapes.append(dt[1]['class_labels'].size(dim=1))
    buckets = np.unique(
      np.percentile(
            shapes,
            np.linspace(0, 100, num_buckets + 1),
            interpolation="lower",
      )[1:]
    )
    return buckets

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  if args.pad:
      encoding = processor.pad(pixel_values, return_tensors="pt", pad_size={"height": 1333, "width": 1333})
  else:
      encoding = processor.pad(pixel_values, return_tensors="pt")
  
  labels = [item[1] for item in batch]
  if args.num_buckets:
    for item in labels:
      num_of_objects = len(item['class_labels'])
      bucketed_size = buckets[np.min(np.where(buckets>=num_of_objects))] 
      idx_to_copy = 0
      for i in range(num_of_objects, bucketed_size):
        item['class_labels'] = torch.cat((item['class_labels'][:], torch.as_tensor([get_num_labels()])))
        item['boxes'] = torch.cat((item['boxes'][:],item['boxes'][0:1]))
        item['area'] = torch.cat((item['area'][:],item['area'][0:1]))
        item['iscrowd'] = torch.cat((item['iscrowd'][:],item['iscrowd'][0:1]))
  
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

train_dataset = CocoDetection(img_folder=img_folder, ann_file=train_ann_file, processor=processor)
val_dataset = CocoDetection(img_folder=img_folder, ann_file=val_ann_file, processor=processor, train=False)

# Prepare bucketing functionality
tmp_dl = DataLoader(train_dataset, batch_size=1, num_workers=15)

def prepare_dataloaders():
    global buckets
    global train_dataloader
    global val_dataloader
    shuffle = False if args.deterministic else True
    buckets = bucketer(tmp_dl, args.num_buckets) if args.num_buckets else None
    print(f'buckets = {buckets}')

    # Dataloaders to be used in training / testing
    # Note, "batch_size=4, num_workers=1" hangs on CPU
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=shuffle)
    val_dataloader   = DataLoader(val_dataset,   collate_fn=collate_fn, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    return train_dataloader, val_dataloader
