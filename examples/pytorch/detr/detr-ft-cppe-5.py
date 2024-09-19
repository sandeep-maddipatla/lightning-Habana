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
from lightning.pytorch.callbacks import ModelCheckpoint
from detr_module import Detr

### Process arguments and prepare dataloaders based on the args
parse_arguments()
show_arguments()
if args.deterministic:
    seed_everything(args.seed, workers=True)
if args.device == 'hpu':
    from lightning_habana.pytorch.profiler.profiler import HPUProfiler
    from lightning_habana.pytorch.accelerator       import HPUAccelerator
    from lightning_habana.pytorch.plugins.precision import HPUPrecisionPlugin
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

if not args.autocast:
    print(f'Autocast is disabled. Casting model to {precision}')
    model.model = model.model.to(precision)

## Debug: model op dynamicity study
## TODO: Enabling this crashes the app. Fix it.
## model.model = detect_recompilation_auto_model(model.model)

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
    'gradient_clip_val' : 0.1,
    'log_every_n_steps' : 1,
    'deterministic' : args.deterministic,
}
trainer_kwargs.update({'callbacks' : [checkpoint_callback]} if args.ckpt_store_interval_epochs != 0 else {'enable_checkpointing' : False} )

if not args.inference_only:
    if args.device == 'cpu' :
        with torch.autocast(device_type="cpu", dtype=precision, enabled=args.autocast):
            trainer = Trainer(
                accelerator='cpu', 
                **trainer_kwargs
                )
    elif args.device == 'hpu':
        with torch.autocast(device_type="hpu", dtype=precision, enabled=args.autocast):
            trainer = Trainer(
                accelerator=HPUAccelerator(),
                **trainer_kwargs
                #plugins=[HPUPrecisionPlugin(precision="32-true")] #bf16-mixed
                #,profiler=HPUProfiler()
            )
    elif args.device == 'cuda':
        print('Warning: CUDA path is untested')
        if args.deterministic:
            torch.use_deterministic_algorithms(True)
        with torch.autocast(device_type="cuda", dtype=precision, enabled=args.autocast):
            trainer = Trainer(
                    accelerator="gpu",
                    devices=1,
                    **trainer_kwargs)
    else:
        print(f'Unsupported device {args.device}')
        exit()

    trainer.fit(model)
    trainer.save_checkpoint("./cppe-5.ckpt")

    # Generate checkpoint reusable with HF API's i.e. AutoModel.from_pretrained() with custom local path instead of HF-Hub. 
    # AutoModel.from_pretrained requires a checkpoint that is generated with save_pretrained.
    # Known Issue: This works only in Lazy Mode.
    try:
        model.model.save_pretrained("./cppe5-hftransformers-ckpt.pt")
    except ValueError as e:
        print('Exception trying to save checkpoint with HF-API save_pretrained. Skipping this step')
        print('Error reported: ' + str(e))

## For model op dynamicity study
## model.model.analyse_dynamicity()

if args.train_only:
    exit()

### Try inference with new weights
from PIL import Image

device = args.device
cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

image_id = 9
pixel_values, target = val_dataset[image_id]
pixel_values = pixel_values.unsqueeze(0)
target_list = [target] #Inference function expects a list of labels for a batch

if not args.autocast:
    model = model.to(precision).to(device)
    pixel_values = pixel_values.to(precision).to(device)

if device == 'cpu':
    print(f'Running Inference with CPU. Precision = {precision}, autocast = {args.autocast}')
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=precision, enabled=args.autocast):
        outputs = model(pixel_values=pixel_values, pixel_mask=None, labels=target_list)
elif device == 'hpu':
    print(f'Running Inference with HPU. Precision = {precision}, autocast = {args.autocast}')
    with torch.no_grad(), torch.autocast(device_type="hpu", dtype=precision, enabled=args.autocast):
        outputs = model(pixel_values=pixel_values, pixel_mask=None, labels=target_list)
        torch.hpu.synchronize()
elif device == 'cuda':
    print(f'Running Inference with CUDA. Precision = {precision}, autocast = {args.autocast}')
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=precision, enabled=args.autocast):
        outputs = model(pixel_values=pixel_values, pixel_mask=None, labels=target_list)
        torch.cuda.synchronize()

#print logits
num_of_objects_to_print = 20 #use 100 to print all
logits = outputs.logits
loss = outputs.loss
print(f'output loss = {loss}')
print("output logits ", logits.shape)
for i in range(0, num_of_objects_to_print):
    row = logits[0][i]
    print(f"    {i:{3}d} ", end="")
    for item in row :
        print(f"{item:>{5}.{2}f} ", end="")
    print("")

id2label[0]="na"  #to avoid error

# Post Process results - save annotated image and collect metrics
image_id = target['image_id'].item()
imageParams = val_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join(img_folder, imageParams['file_name']))

output_batch = [outputs]  # List of batch-results. Each entry is model output for one batch of input images submitted to model.
target_batch = [[target]] # List of List of target results per image. Each Inner list corresponds to a batch submitted to model
image_list = [[image]]    # List of List of input images. Each Inner list corresponds to a batch submitted to model

metrics = post_process_model_outputs(output_batch, target_batch, processor, threshold=args.threshold, id2label=id2label, image_batch_list=image_list)
print(f'metrics = {metrics}')
