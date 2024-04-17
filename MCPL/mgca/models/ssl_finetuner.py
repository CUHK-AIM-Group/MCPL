import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from torchmetrics import AUROC, Accuracy
from pytorch_lightning import LightningModule
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
# from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC

class SSLFineTuner(LightningModule):
    def __init__(self,
                 backbone: nn.Module,
                 in_features: int = 2048,
                 num_classes: int = 14,
                 hidden_dim: Optional[int] = 512,
                 dropout: float = 0.0,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-6,
                 multilabel: bool = True,
                 name: str = "clip",
                 *args,
                 **kwargs
                 ):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        if multilabel:
            self.train_auc = AUROC(task='multilabel', num_labels=num_classes)
            self.val_auc = AUROC(task='multilabel', num_labels=num_classes, compute_on_step=False)
            self.test_auc = AUROC(task='multilabel', num_labels=num_classes, compute_on_step=False)
        
            self.train_acc = Accuracy(task='multilabel', num_labels=num_classes, topk=1)
            self.val_acc = Accuracy(task='multilabel', num_labels=num_classes, topk=1, compute_on_step=False)
            self.test_acc = Accuracy(task='multilabel', num_labels=num_classes, topk=1, compute_on_step=False)
        else:
            self.train_auc = AUROC(task='multiclass', num_classes=num_classes)
            self.val_auc = AUROC(task='multiclass', num_classes=num_classes, compute_on_step=False)
            self.test_auc = AUROC(task='multiclass', num_classes=num_classes, compute_on_step=False)
        
            self.train_acc = Accuracy(task='multiclass', num_classes=num_classes, topk=1)
            self.val_acc = Accuracy(task='multiclass', num_classes=num_classes, topk=1, compute_on_step=False)
            self.test_acc = Accuracy(task='multiclass', num_classes=num_classes, topk=1, compute_on_step=False)

        self.name = name
        self.backbone = backbone
        if self.name == "medclip":
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif self.name == "biomedclip":
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif self.name == "mgca":
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif self.name == "imagenet":
            for param in self.backbone.parameters():
                param.requires_grad = False
            # for name, param in self.backbone.named_parameters():
            #     print(name)
        
        self.class_layer = SSLEvaluator(n_input=in_features, n_classes=num_classes, p=dropout, n_hidden=hidden_dim)
        self.multilabel = multilabel

    def training_step(self, batch, batch_idx):
        loss, logit, y = self.shared_step(batch)
        if self.multilabel:
            ys = torch.cat([y, torch.ones(1, 14).cuda()], dim=0)
            logits = torch.cat([logit, torch.ones(1, 14).cuda()], dim=0)
            acc = self.train_acc(torch.sigmoid(logits), ys.long())
            auc = self.train_auc(torch.sigmoid(logits), ys.long())
        else:
            acc = self.train_acc(F.softmax(logit, dim=-1), y.long())
            auc = self.train_auc(F.softmax(logit, dim=-1), y.long())
        
        log = {"train_loss": loss, "train_ACC": acc, "train_AUC": auc}
        self.log_dict(log, sync_dist=True, prog_bar=True)
        if batch_idx % 50 == 0:
            print('Train Batch: [%d] | Loss: %.4f | ACC: %.4f | AUC: %.4f' % (batch_idx, loss, acc, auc) )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logit, y = self.shared_step(batch)
        if self.multilabel:
            ys = torch.cat([y, torch.ones(1, 14).cuda()], dim=0)
            logits = torch.cat([logit, torch.ones(1, 14).cuda()], dim=0)
            acc = self.val_acc(torch.sigmoid(logits), ys.long())
            auc = self.val_auc(torch.sigmoid(logits), ys.long())
        else:
            acc = self.val_acc(F.softmax(logit, dim=-1), y.long())
            auc = self.val_auc(F.softmax(logit, dim=-1), y.long())

        log = {"val_loss": loss, "val_ACC": acc, "val_AUC": auc}
        self.log_dict(log, sync_dist=True, prog_bar=True)
        if batch_idx % 50 == 0:
            print('Val Batch: [%d] | Loss: %.4f | ACC: %.4f | AUC: %.4f' % (batch_idx, loss, acc, auc) )
        return loss

    def test_step(self, batch, batch_idx):
        loss, logit, y = self.shared_step(batch)
        if self.multilabel:
            ys = torch.cat([y, torch.ones(1, 14).cuda()], dim=0)
            logits = torch.cat([logit, torch.ones(1, 14).cuda()], dim=0)
            acc = self.test_acc(torch.sigmoid(logits), ys.long())
            auc = self.test_auc(torch.sigmoid(logits), ys.long())
        else:
            acc = self.test_acc(F.softmax(logit, dim=-1), y.long())
            auc = self.test_auc(F.softmax(logit, dim=-1), y.long())
        
        log = {"Test_loss": loss, "test_ACC": acc, "test_AUC": auc}
        self.log_dict(log, sync_dist=True, prog_bar=True)
        print('Test Batch: [%d] | Loss: %.4f | ACC: %.4f | AUC: %.4f' % (batch_idx, loss, acc, auc) )
        return loss

    def shared_step(self, batch):
        images, gts = batch
        # For multi-class
        with torch.no_grad():
            if self.name == "mgca":
                feats, _ = self.backbone(images)   # MGCA
            else:
                feats = self.backbone(images)      # CLIP MedCLIP BiomedCLIP
        
        feats = feats.view(feats.size(0), -1)
        logits = self.class_layer(feats.float())

        if self.multilabel:
            loss = F.binary_cross_entropy_with_logits(logits, gts.float())
        else:
            gts = gts.squeeze()
            loss = F.cross_entropy(logits, gts.long())
        return loss, logits, gts


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.class_layer.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay)
        # return optimizer
        
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.learning_rate,
            min_lr=1e-6,
            warmup_steps=int(self.training_steps * 0.2))
        
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        effective_batch_size = trainer.accumulate_grad_batches
        return (dataset_size // effective_batch_size) * trainer.max_epochs


class SSLEvaluator(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=None, p=0.0) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if self.n_hidden is None:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes) )
        else:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes) )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class Flatten(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)




