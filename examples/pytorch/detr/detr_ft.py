import os
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader

from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import DataLoader

import lightning as pl
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

import argparse
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

#### Init variables for this script.

# Args
args = argparse.Namespace()

# Dataset related variables
num_labels = 5  #TODO: automate this from json files in dataset instead of hardcoding
buckets = None
train_dataloader = None
val_dataloader = None
train_dataset = None
val_dataset = None
img_folder = '/root/gs-274/CPPE-Dataset/data/images'
annotations_folder = '/root/gs-274/CPPE-Dataset/data/annotations'
train_ann_file = os.path.join(annotations_folder, "train.json")
val_ann_file = os.path.join(annotations_folder, "test.json")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Variables for Post-processing outputs
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]    

# Argument Parsers
def parse_arguments():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference-only', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--autocast', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use-ckpt', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--max-epochs', default=5, type=int)
    parser.add_argument('--max-steps', default=100000, type=int)
    parser.add_argument('--num-workers', default=15, type=int)
    parser.add_argument('--deterministic', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--train-only', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--pad', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--num-buckets', default=1, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', choices=['hpu', 'cpu', 'cuda'], default='hpu', type=str)
    parser.add_argument('--ckpt-path', default="./cppe-5.ckpt", type=str)
    parser.add_argument('--ckpt-store-interval-epochs', default=0, type=int)
    parser.add_argument('--ckpt-store-path', default="./", type=str)
    
    # Arguments specifically for inference
    parser.add_argument('--max-inf-frames', default=0, type=int)
    parser.parse_args(namespace=args)

    # Do basic validation of args and adjust if required
    if args.device not in ["cpu", "hpu", "cuda"]:
        print(f'Unsupported device {args.device}')
        exit()
    if args.deterministic:
        args.num_workers = 0

def show_arguments():
    print(f'inference-only = {args.inference_only}')
    print(f'autocast = {args.autocast}')
    print(f'use-ckpt = {args.use_ckpt}')
    print(f'batch-size = {args.batch_size}')
    print(f'max-epochs = {args.max_epochs}')
    print(f'max-steps = {args.max_steps}')
    print(f'num-workers = {args.num_workers}')
    print(f'deterministic = {args.deterministic}')
    print(f'train-only = {args.train_only}')
    print(f'pad = {args.pad}')
    print(f'num-buckets = {args.num_buckets}')
    print(f'threshold = {args.threshold}')
    print(f'seed = {args.seed}')
    print(f'device = {args.device}')
    print(f'ckpt-path = {args.ckpt_path}')
    print(f'ckpt-store-interval-epochs = {args.ckpt_store_interval_epochs}')
    print(f'ckpt-store-path = {args.ckpt_store_path}')
    print(f'max-inf-frames = {args.max_inf_frames}')
    
    # Derived parameters
    shuffle = False if args.deterministic else True
    precision = torch.bfloat16 if args.autocast else torch.float32
    print(f'Derived params: precision = {precision}, shuffle = {shuffle}')

# Classes and functions to process dataset
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

def init_datasets():
    train_dataset = CocoDetection(img_folder=img_folder, ann_file=train_ann_file, processor=processor)
    val_dataset = CocoDetection(img_folder=img_folder, ann_file=val_ann_file, processor=processor, train=False)
    return train_dataset, val_dataset

def prepare_dataloaders(train_dataset, val_dataset, test_dataset=None):
    global buckets
    global train_dataloader
    global val_dataloader
    shuffle = False if args.deterministic else True

    # Handle bucketing logic
    tmp_dl = DataLoader(train_dataset, batch_size=1, num_workers=15)
    buckets = bucketer(tmp_dl, args.num_buckets) if args.num_buckets else None
    print(f'buckets = {buckets}')

    # Dataloaders to be used in training / testing
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=shuffle)
    val_dataloader   = DataLoader(val_dataset,   collate_fn=collate_fn, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    return train_dataloader, val_dataloader

# Detr class based on lightning

class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, train_dl, val_dl):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                             revision="no_timm",
                                                             num_labels=get_num_labels(),
                                                             ignore_mismatched_sizes=True)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.train_dl = train_dl
        self.val_dl = val_dl

    def forward(self, pixel_values, pixel_mask=None, labels=None):
        if labels is not None:
            labels = [{k: d[k].to(self.device) for k in d.keys()} for d in labels]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        return outputs

    def predict_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = batch["labels"]
        return self(pixel_values, pixel_mask, labels)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        print(f'loss={loss}')
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        print(f'loss={loss}')
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())
        return loss

    def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)
        return optimizer

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

# Post-Processing helper function
def plot_results(pil_img, scores, labels, boxes, tag="", id2label=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig('output_image' + tag + '.jpg')
    print(f"Saved output image as {'output_image' + tag + '.jpg'}")


def main():
    ### Run Training and test sample result

    ##  Step-1) Parse args and prepare configuration controls
    parse_arguments()
    show_arguments()
    if args.deterministic:
        seed_everything(args.seed, workers=True)
    if args.device == 'hpu':
        from lightning_habana.pytorch.accelerator       import HPUAccelerator
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
        adapt_transformers_to_gaudi()

    # MixedPrecision requires a lower precision type specified
    # Use BFloat16 when autocast is enabled.
    precision = torch.bfloat16 if args.autocast else torch.float32

    ##  Step-2) Prepare dataloaders based on the args
    train_dataset, val_dataset = init_datasets()
    train_dl, val_dl = prepare_dataloaders(train_dataset, val_dataset)

    ##  Step-3) Load model with specified configuration
    if args.use_ckpt:
        try:
            #load model from lightning checkpoint
            model = Detr.load_from_checkpoint(args.ckpt_path, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train_dl=train_dl, val_dl=val_dl)
        except Exception as e:
            print(f'Error attempting to load checkpoint at {args.ckpt_path} .. Reverting to default model')
            print(f'Error message: {str(e)}')
            #load default model
            model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train_dl=train_dl, val_dl=val_dl)
    else:
        #load default model
        model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train_dl=train_dl, val_dl=val_dl)

    ##  Step-4) Prepare trainer object and call trainer.fit, save checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath = args.ckpt_store_path,
        every_n_epochs = args.ckpt_store_interval_epochs,
        filename = 'detr-cppe5-epoch{epoch:03d}',
        auto_insert_metric_name=False,
        save_top_k = -1
    )
    trainer_kwargs = {
        'max_steps' : args.max_steps,
        'max_epochs' : args.max_epochs,
        'devices' : 1,
        'gradient_clip_val' : 0.1,
        'log_every_n_steps' : 1,
        'deterministic' : args.deterministic,
    }
    trainer_kwargs.update({'callbacks' : [checkpoint_callback]} if args.ckpt_store_interval_epochs != 0 else {'enable_checkpointing' : False} )

    trainer_acc = args.device
    if args.device == "cuda":
        print('Warning: CUDA path is untested')
        if args.deterministic:
            torch.use_deterministic_algorithms(True)
        trainer_acc = "cuda"

    if not args.autocast:
        model = model.to(precision).to(args.device)

    if not args.inference_only:
        with torch.autocast(device_type=args.device, dtype=precision, enabled=args.autocast):
            trainer = Trainer(accelerator = trainer_acc, **trainer_kwargs)
            trainer.fit(model)
            trainer.save_checkpoint("./cppe-5.ckpt")

    if args.train_only:
        exit()

    ##  Step-5) Run inference on one image and save annotated output

    ### Try inference with new weights
    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k,v in cats.items()}
    id2label[0]="na"  #to avoid error

    image_id = 9 # Picking 9th image in val dataset
    pixel_values, target = val_dataset[image_id]
    pixel_values = pixel_values.unsqueeze(0)

    print(f'Running Inference with Device {args.device}. Precision = {precision}, autocast = {args.autocast}')
    with torch.no_grad(), torch.autocast(device_type=args.device, dtype=precision, enabled=args.autocast):
        outputs = model(pixel_values=pixel_values, pixel_mask=None)

    # Post Process results - save annotated image
    image_size = torch.tensor([target["orig_size"].numpy()])
    post_processed_outputs = processor.post_process_object_detection(outputs, threshold=args.threshold, target_sizes=image_size)
    for results in post_processed_outputs:
        # Only one result (for one image) expected in list of outputs
        image_id = target['image_id'].item()
        image_params = val_dataset.coco.loadImgs(image_id)[0]
        image = Image.open(os.path.join(img_folder, image_params['file_name']))
        plot_results(image, results['scores'], results['labels'], results['boxes'], tag=str(image_id), id2label=id2label)

if __name__ == '__main__':
    main()