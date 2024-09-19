import torch
import lightning as pl
from transformers import DetrForObjectDetection
from dataset_cppe5 import get_num_labels

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
        
        print(f"\ndetr-ft-cppe-5.py::Detr::training_step loss({loss:{6}.{2}f})\n")
       
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