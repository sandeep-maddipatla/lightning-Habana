import os
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader

from arguments import parse_arguments, show_arguments, args
from dataset_cppe5 import processor, prepare_dataloaders, get_num_labels, img_folder, train_dataset, val_dataset
from evaluate import post_process_model_outputs

import lightning as pl
from lightning import Trainer, seed_everything

from detr_module import Detr
import matplotlib.pyplot as plt
from PIL import Image

parse_arguments()
show_arguments()
if args.deterministic:
    seed_everything(args.seed, workers=True)
if args.device == 'hpu':
    from lightning_habana.pytorch.profiler.profiler import HPUProfiler
    from lightning_habana.pytorch.accelerator       import HPUAccelerator
    from lightning_habana.pytorch.plugins.precision import HPUPrecisionPlugin
    from habana_frameworks.torch.utils.experimental import data_dynamicity
    from habana_frameworks.torch.utils.experimental import detect_recompilation_auto_model
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    adapt_transformers_to_gaudi()

train_dl, val_dl = prepare_dataloaders()
precision = torch.bfloat16 if args.bf16 else torch.float32

if args.use_ckpt:
    try:
        #load model from checkpoint
        model = Detr.load_from_checkpoint(args.ckpt_path, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train_dl=train_dl, val_dl=val_dl)

        # Uncomment below if we need to convert the lightning checkpoint to one that is reloadable in plain HF-transformers API's i.e.
        # AutoModel.from_pretrained() with custom local path instead of HF-Hub. 
        # AutoModel.from_pretrained requires a checkpoint that is generated with save_pretrained.
        #model.model.save_pretrained("./cppe5-hftransformers-ckpt.pt")
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
        
cats = val_dataset.coco.cats
print(f'cats = {cats}')
id2label = {k: v['name'] for k,v in cats.items()}
print(f'id2label = {id2label}')

# Inferencing
if args.device == 'cpu':
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=precision, enabled=args.autocast):
        trainer = Trainer(
                accelerator='cpu', max_steps=args.max_steps, max_epochs=args.max_epochs, gradient_clip_val=0.1, enable_checkpointing=False, log_every_n_steps=1,
                deterministic=args.deterministic
                )
elif args.device == 'hpu':
    with torch.no_grad(), torch.autocast(device_type="hpu", dtype=precision, enabled=args.autocast):
        trainer = Trainer(
                accelerator=HPUAccelerator(),max_steps=args.max_steps, max_epochs=args.max_epochs, gradient_clip_val=0.1, enable_checkpointing=False, log_every_n_steps=1,
                deterministic = args.deterministic,
                #plugins=[HPUPrecisionPlugin(precision="32-true")] #bf16-mixed
                #,profiler=HPUProfiler()
            )
elif args.device == 'cuda':
    print('CUDA path is unverified')
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=precision, enabled=args.autocast):
        trainer = Trainer(
                accelerator="gpu",
                max_steps=args.max_steps, max_epochs=args.max_epochs, gradient_clip_val=0.1, enable_checkpointing=False, log_every_n_steps=1,
                deterministic = args.deterministic,
                devices=1
            )
else:
    print(f'Unsupported device {args.device}')
    exit()

autocast = torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True)

with torch.no_grad(), autocast:
    # forward pass to get class logits and bounding boxes
    predictions = trainer.predict(model, dataloaders = val_dl)
    if args.device == 'hpu':
        torch.hpu.synchronize()
    elif args.device == 'cuda':
        torch.cuda.synchronize()
  
# Prepare targets and images
target_batch_list = []
image_batch_list = []

def end_test(batch_count):
    if (args.max_inf_frames > 0) and (args.batch_size * batch_count >= args.max_inf_frames):
        return True
    else:
        return False

val_dl_enum = enumerate(val_dl)
step = 0
while not end_test(step):
    try:
        step, input = next(val_dl_enum)
        pv = input["pixel_values"]
        target_batch = input["labels"]
    except StopIteration:
        break

    target_batch_list.append(target_batch)
    image_batch = []
    for target in target_batch:
        image_id = target['image_id'].item()
        image = val_dataset.coco.loadImgs(image_id)[0]
        image = Image.open(os.path.join(img_folder, image['file_name']))
        image_batch.append(image)
    image_batch_list.append(image_batch)


# postprocess model outputs
metrics = post_process_model_outputs(predictions, target_batch_list, processor, threshold=args.threshold, id2label=id2label, image_batch_list=image_batch_list)
print(f'metrics = {metrics}')

