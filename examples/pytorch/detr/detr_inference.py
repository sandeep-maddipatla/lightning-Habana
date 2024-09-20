import os
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader

import lightning as pl
from lightning import Trainer, seed_everything

import matplotlib.pyplot as plt
from PIL import Image
from detr_ft import parse_arguments, show_arguments, args, train_dataset, val_dataset, Detr, processor, plot_results

def end_pp_loop(batch_count):
    if (args.max_inf_frames > 0) and (args.batch_size * batch_count >= args.max_inf_frames):
        return True
    else:
        return False

def synchronize():
    if args.device == 'hpu':
        torch.hpu.synchronize()
    elif args.device == 'cuda':
        torch.cuda.synchronize()
asdfasdf
### Run inference over validation dataset

## Step-1: Parse args and set up config
parse_arguments()
show_arguments()
if args.deterministic:
    seed_everything(args.seed, workers=True)
if args.device == 'hpu':
    from lightning_habana.pytorch.accelerator       import HPUAccelerator
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    adapt_transformers_to_gaudi()
precision = torch.bfloat16 if args.bf16 else torch.float32

## Step-2: Load model with specified checkpoint or defaults if none is specified
if args.use_ckpt:
    try:
        #load model from checkpoint
        model = Detr.load_from_checkpoint(args.ckpt_path, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train_dl=train_dl, val_dl=val_dl)
    except Exception as e:
        print(f'Error attempting to load checkpoint at {args.ckpt_path} .. Reverting to default model')
        print(f'Error message: {str(e)}')
        #load default model
        model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train_dl=train_dl, val_dl=val_dl)
else:
    #load default model
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train_dl=train_dl, val_dl=val_dl)

model = model.eval().to(args.device)
if not args.autocast:
    print(f'Autocast is disabled. Casting model to {precision}')
    model.model = model.model.to(precision)       
        
## Step-3: Prepare trainer and run inference
trainer_kwargs = {
    'max_steps' : args.max_steps,
    'max_epochs' : args.max_epochs,
    'devices' : 1,
    'gradient_clip_val' : 0.1,
    'log_every_n_steps' : 1,
    'deterministic' : args.deterministic,
    'enable_checkpointing' : False,
}
trainer_kwargs.update({'callbacks' : [checkpoint_callback]} if args.ckpt_store_interval_epochs != 0 else {'enable_checkpointing' : False} )

trainer_acc = args.device
if args.device == "cuda":
    print('Warning: CUDA path is untested')
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    trainer_acc = "cuda"

print(f'Running Inference with Device {args.device}. Precision = {precision}, autocast = {args.autocast}')
with torch.no_grad(), torch.autocast(device_type=args.device, dtype=precision, enabled=args.autocast):
    trainer = Trainer(accelerator = trainer_acc, **trainer_kwargs)
    # forward pass to get class logits and bounding boxes
    predictions = trainer.predict(model, dataloaders = val_dl)
    synchronize()

## Step-4: Annotate and save output images
cats = val_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}
val_dl_enum = enumerate(val_dl)

image_batch_list = []
image_sizes = []
step = 0

# Prepare image batches and image-sizes lists (one entry of size batch_size per batch)
while not end_pp_loop(step):
    try:
        step, input = next(val_dl_enum)
        pv = input["pixel_values"]
        target_batch = input["labels"]
    except StopIteration:
        break
    
    batch_image_sizes = torch.tensor([x["orig_size"].numpy() for x in target_batch])
    image_sizes.append(batch_image_sizes) 
    
    image_batch = []
    for target in target_batch:
        image_id = target['image_id'].item()
        image = val_dataset.coco.loadImgs(image_id)[0]
        image = Image.open(os.path.join(img_folder, image['file_name']))
        image_batch.append(image)
    image_batch_list.append(image_batch)

# Run through Prediction-list (one entry of size batch_size per batch), with image_sizes and images, and post-process
for batch, target_sizes, image_batch in zip(predictions, image_sizes, image_batch_list):
    post_processed_output = processor.post_process_object_detection(
                            batch, threshold=threshold, target_sizes=target_sizes
                        )
    # Draw annotated image and save it as a file
    try:
        for results, image in zip(post_processed_output, image_batch):
            plot_results(image, results['scores'], results['labels'], results['boxes'], id2label=id2label, tag=str(count))
            count += 1
    except Exception as e:
        print(f'Exception in saving annotated image at count={count}. Skipping')
        print(f'Error message: {str(e)}')