import os
import math
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from dateutil import tz
from argparse import ArgumentParser
from torchmetrics import AUROC, Accuracy
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)

from mgca.datasets.data_module import DataModule
from mgca.datasets.transforms import DataTransforms
from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset, multimodal_collate_fn)

# from mgca_old import MGCA

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from mcpl import MGCA
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

JP_list = ['pleural', 'lung', 'pulmonary', 'lungs', 'chest', 'silhouette', 'mediastinal', 
           'cardiomediastinal', 'heart', 'hilar', 'tube', 'osseous', 'lobe', 'vascular', 
           'thoracic', 'catheter', 'interval', 'bibasilar', 'aorta', 'vasculature', 'interstitial', 
           'svc', 'spine', 'silhouettes', 'rib', 'otracheal', 'bony', 'sternotomy', 
           'stomach', 'retrocardiac', 'aortic', 'basilar', 'picc', 'clips', 'costophrenic', 
           'abdomen', 'atrium', 'wires', 'venous', 'nasogastric', 'fluid', 'ventricle', 
           'pacemaker', 'jugular', 'bronchovascular', 'vascularity', 'enteric', 'hila', 'diaphragm', 
           'perihilar', 'port-a-cath', 'arch', 'hemithorax', 'subclavian', 'tissue', 'cavoatrial', 
           'knob', 'vertebral', 'tracheostomy', 'valve', 'pacer', 'artery', 'hiatal', 
           'trachea', 'vein', 'cabg', 'subcutaneous', 'tubes', 'esophagus', 'stent', 
           'vessels', 'cervical', 'sternal', 'neck', 'junction']                # 75

BL_list = ['effusion', 'pneumothorax', 'consolidation', 'focal', 'cardiac', 'atelectasis', 'edema', 
           'opacity', 'effusions', 'opacities', 'pneumonia', 'congestion', 'heiaphragm', 'cardiomegaly', 
           'carina', 'opacification', 'degenerative', 'fracture', 'fractures', 'chronic', 'mediastinum', 
           'calcifications', 'infection', 'disease', 'emphysema', 'tortuosity', 'calcification', 'consolidations', 
           'calcified', 'thickening', 'parenchymal', 'atherosclerotic', 'nodular',  'hernia', 'deformity', 
           'engorgement', 'collapse', 'nodule', 'multifocal', 'infectious', 'pneumothoraces', 'density', 
           'diffuse', 'streaky']                                                # 44


# 构建CLIP模型
def load_clip_to_cpu():
    backbone_name = "ViT-B/16"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)  
    try:  # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe', "vision_depth": 0, "language_depth": 0, "vision_ctx": 0, "language_ctx": 0, "maple_length": 2}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    
    model = model.eval() 
    for p in model.parameters(): # 冻结clip的参数
     	p.requires_grad = False
    return model


# 文本编码器
class text_encoder(nn.Module):
    def __init__(self, clip_model, model):
        super().__init__()

        self.transformer = model.text_encoder.transformer           # 读取transformer模型
        self.positional_embedding = clip_model.positional_embedding # 位置编码
        self.ln_final = clip_model.ln_final                         # layer norm
        self.text_projection = clip_model.text_projection           # 映射层
        self.dtype = clip_model.dtype                               # 数据类型

    def forward(self, texts, tokenized_prompts, JP_prompts, BL_prompts, compound_prompts_text, texts_JP, texts_BL):
        
        x = texts + self.positional_embedding.type(self.dtype)      # 文本 + 位置编码
        x = x.permute(1, 0, 2)                                      # NLD -> LND
        x1 = texts_JP + self.positional_embedding.type(self.dtype)
        x2 = texts_BL + self.positional_embedding.type(self.dtype)
        x1 = x1.permute(1, 0, 2)
        x2 = x2.permute(1, 0, 2)
        combined = [x, compound_prompts_text, 0, x1, x2, JP_prompts, BL_prompts, torch.empty(x.shape[1], 6, 154)]     # 文本与提示组成输入
        outputs = self.transformer(combined)
        x = outputs[0]                                              # 提取输出的第一个向量
        x = x.permute(1, 0, 2)                                      # LND -> NLD
        x = self.ln_final(x).type(self.dtype)                       # batchsize tokensize dim 8 77 512
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection # 选出x每行的最大值 8 512
        return x


# 分类器
class Flatten(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)
    
class ml_classer(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=None, p=0.1) -> None:
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


# 多模态提示
class mm_prompter(nn.Module):
    def __init__(self, clip_model, model, resnet):
        super().__init__()
        
        self.n_cls = 256   # batch size
        self.dtype = model.prompt_learner.dtype
        self.token_embed = model.prompt_learner.token_embed
        
        self.RS = resnet.eval()
        
        self.ctx_text = model.prompt_learner.ctx_text
        self.ctx_img = model.prompt_learner.ctx_img                
        
        self.proj_JB = model.prompt_learner.proj_JB

        self.compound_prompts_text = model.prompt_learner.compound_prompts_text
        self.compound_prompts_img = model.prompt_learner.compound_prompts_img
        self.JP_prompts = model.prompt_learner.JP_prompts
        self.BL_prompts = model.prompt_learner.BL_prompts

        self.MSA_1 = model.prompt_learner.MSA_1
        self.MSA_2 = model.prompt_learner.MSA_2
        self.MSA_3 = model.prompt_learner.MSA_3

    def forward(self, batch):

        ctx_text = self.ctx_text
        if ctx_text.dim() == 2:
            ctx_text = ctx_text.unsqueeze(0).expand(self.n_cls, -1, -1)
        ctx_img = self.ctx_img
        
        texts = self.token_embed(batch["tokens"]).type(self.dtype) # batch 77 512
        prefix = texts[:, :1, :]
        suffix = texts[:, 5:77, :]
        texts = torch.cat([prefix, ctx_text, suffix], dim=1) # 提示插入文本特征
        
        texts_JP = self.token_embed(batch["tokens_JP"]).type(self.dtype).cuda()
        texts_BL = self.token_embed(batch["tokens_BL"]).type(self.dtype).cuda()
        
        ''''''
        labels_JP = batch["labels_JP"]
        labels_BL = batch["labels_BL"]
        att_T = torch.cat([labels_JP, labels_BL], dim=1)
        
        ''''''
        img = batch["imgs"].cuda()
        att_I = self.RS(img)
        att_I = torch.sigmoid(att_I)
        lb_JP = att_I[:, 0:75]
        lb_BL = att_I[:, 76:120]
        
        token_JP = []
        for i in range(lb_JP.shape[0]):
            sent_JP = [JP_list[pos] for pos, num in enumerate(lb_JP[i,:]) if num == 1.0] 
            token_JP.append(clip.tokenize(' '.join(sent_JP), truncate=True))   
        token_JP = torch.stack(token_JP).squeeze().cuda()    
        embed_JP = self.token_embed(token_JP).type(self.dtype)
        
        token_BL = []
        for i in range(lb_BL.shape[0]):
            sent_BL = [BL_list[pos] for pos, num in enumerate(lb_BL[i,:]) if num == 1.0] 
            token_BL.append(clip.tokenize(' '.join(sent_BL), truncate=True))   
        token_BL = torch.stack(token_BL).squeeze().cuda()    
        embed_BL = self.token_embed(token_BL).type(self.dtype)
        
        ''''''
        # imgs_JP = self.proj_JB(texts_JP) # 文本转换为图像特征
        # imgs_BL = self.proj_JB(texts_BL)
        
        imgs_JP = self.proj_JB(embed_JP) # 文本转换为图像特征
        imgs_BL = self.proj_JB(embed_BL)
        
        # 文本提示和视觉提示交互
        textual_prompts = []     # 6layer, 8, 512dim
        visual_prompts = []      # 6layer, 8, 768dim
        
        joint_in_1 = torch.cat([self.compound_prompts_text[0], self.compound_prompts_img[0]], dim=0).unsqueeze(0)
        joint_out_1 = self.MSA_1(joint_in_1)
        joint_out_1 = joint_out_1.squeeze(0)
        textual_prompts.append(joint_out_1[0:2, 0:512])
        visual_prompts.append(joint_out_1[2:4, :])
        
        joint_in_2 = torch.cat([self.compound_prompts_text[1], self.compound_prompts_img[1]], dim=0).unsqueeze(0)
        joint_out_2 = self.MSA_2(joint_in_2)
        joint_out_2 = joint_out_2.squeeze(0)
        textual_prompts.append(joint_out_2[0:2, 0:512])
        visual_prompts.append(joint_out_2[2:4, :])
        
        joint_in_3 = torch.cat([self.compound_prompts_text[2], self.compound_prompts_img[2]], dim=0).unsqueeze(0)
        joint_out_3 = self.MSA_3(joint_in_3)
        joint_out_3 = joint_out_3.squeeze(0)
        textual_prompts.append(joint_out_3[0:2, 0:512])
        visual_prompts.append(joint_out_3[2:4, :])
        
        return texts, texts_JP, texts_BL, imgs_JP, imgs_BL, att_T.float(), att_I.float(), ctx_img, self.JP_prompts, self.BL_prompts, textual_prompts, visual_prompts
        #      报告   解剖学词   病理学词  解剖学图  病理学图                               初始视觉提示    解剖学提示       病理学提示   文本提示         视觉提示 


class ClassFineTuner(LightningModule):
    def __init__(self,
                 model: nn.Module,
                 resnet: nn.Module,
                 in_features: int = 512,
                 num_classes: int = 14,
                 dropout: float = 0.0,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-6,
                 multilabel: bool = True,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        # init encoders
        clip_model = load_clip_to_cpu() ##
        
        self.image_encoder = model.image_encoder
        self.text_encoder = text_encoder(clip_model, model)
        self.prompt_learner = mm_prompter(clip_model, model, resnet)
        self.class_layer = ml_classer(n_input=512, n_classes=14, p=0.0, n_hidden=512)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.train_auc = AUROC(task='multilabel', num_labels=14)
        self.val_auc = AUROC(task='multilabel', num_labels=14, compute_on_step=False)
        self.test_auc = AUROC(task='multilabel', num_labels=14, compute_on_step=False)
        
        self.train_acc = Accuracy(task='multilabel', num_labels=14, topk=1)
        self.val_acc = Accuracy(task='multilabel', num_labels=14, topk=1, compute_on_step=False)
        self.test_acc = Accuracy(task='multilabel', num_labels=14, topk=1, compute_on_step=False)

        # self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=121)
        self.multilabel = multilabel

    def training_step(self, batch, batch_idx, split="train"):
        loss, logit, y = self.shared_step(batch)
        
        ys = torch.cat([y, torch.ones(1, 14).cuda()], dim=0)
        logits = torch.cat([logit, torch.ones(1, 14).cuda()], dim=0)
        acc = self.train_acc(torch.sigmoid(logits), ys.long())
        auc = self.train_auc(torch.sigmoid(logits), ys.long())
        
        log = {"train_loss": loss, "train_ACC": acc, "train_AUC": auc}
        self.log_dict(log, sync_dist=True, prog_bar=True)
        if batch_idx % 50 == 0:
            print('Train Batch: [%d] | Loss: %.4f | ACC: %.4f | AUC: %.4f' % (batch_idx, loss, acc, auc) )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logit, y = self.shared_step(batch)
        
        ys = torch.cat([y, torch.ones(1, 14).cuda()], dim=0)
        logits = torch.cat([logit, torch.ones(1, 14).cuda()], dim=0)
        acc = self.val_acc(torch.sigmoid(logits), ys.long())
        auc = self.val_auc(torch.sigmoid(logits), ys.long())
        
        log = {"val_loss": loss, "val_ACC": acc, "val_AUC": auc}
        self.log_dict(log, sync_dist=True, prog_bar=True)
        if batch_idx % 50 == 0:
            print('Val Batch: [%d] | Loss: %.4f | ACC: %.4f | AUC: %.4f' % (batch_idx, loss, acc, auc) )
        return loss

    def test_step(self, batch, batch_idx):
        loss, logit, y = self.shared_step(batch)
        
        ys = torch.cat([y, torch.ones(1, 14).cuda()], dim=0)
        logits = torch.cat([logit, torch.ones(1, 14).cuda()], dim=0)
        acc = self.test_acc(torch.sigmoid(logits), ys.long())
        auc = self.test_auc(torch.sigmoid(logits), ys.long())
        
        log = {"Test_loss": loss, "test_ACC": acc, "test_AUC": auc}
        self.log_dict(log, sync_dist=True, prog_bar=True)
        print('Test Batch: [%d] | Loss: %.4f | ACC: %.4f | AUC: %.4f' % (batch_idx, loss, acc, auc) )
        return loss

    def shared_step(self, batch):
        # tokenized_texts = batch["tokens_77"]
        images = batch["imgs"].half()
        gts = batch["gts"]

        texts, texts_JP, texts_BL, imgs_JP, imgs_BL, _, _, shared_ctx, JP_prompts, BL_prompts, prompts_text, prompts_vision = self.prompt_learner(batch)

        image_features, _ = self.image_encoder(images, shared_ctx, JP_prompts, BL_prompts, prompts_vision, imgs_JP, imgs_BL)
        # text_features = self.text_encoder(texts, tokenized_texts, JP_prompts, BL_prompts, prompts_text, texts_JP, texts_BL)
        
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits = self.class_layer(image_features.float())
        loss = F.binary_cross_entropy_with_logits(logits, gts.float())

        return loss, logits, gts

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.class_layer.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay)

        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.learning_rate,
            min_lr=1e-6,
            warmup_steps=int(self.training_steps * 0.2))
            
        scheduler = {"scheduler": lr_scheduler,
                     "interval": "step",
                     "frequency": 1}
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer


    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        effective_batch_size = trainer.accumulate_grad_batches
        return (dataset_size // effective_batch_size) * trainer.max_epochs

''''''
def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chexpert")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--in_features", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_pct", type=float, default=1.0)
    # add trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # set max epochs
    args.max_epochs = 15

    seed_everything(args.seed)

    num_classes = 14
    multilabel = True

    datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn, DataTransforms, args.data_pct, args.batch_size, args.num_workers)

    args.num_classes = num_classes
    args.multilabel = multilabel

    # Add load from checkpoint
    model = MGCA.load_from_checkpoint('pretrain.ckpt', strict=False)
    for p in model.parameters(): # 冻结clip的参数
        p.requires_grad = False
    args.model = model
    
    resnet = torch.load('RS2.pth')
    for p in resnet.parameters(): # 冻结clip的参数
        p.requires_grad = False
    args.resnet = resnet
    # finetune
    tuner = ClassFineTuner(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(BASE_DIR, f"D:/124/MGCA-main/class/ckpts/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    callbacks = [LearningRateMonitor(logging_interval="step"),
                 ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir, save_last=True, mode="min", save_top_k=5),
                 EarlyStopping(monitor="val_loss", min_delta=0., patience=10, verbose=False, mode="min") ]

    # get current time
    # now = datetime.datetime.now(tz.tzlocal())
    # extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    
    logger_dir = os.path.join(BASE_DIR, f"D:/124/MGCA-main/class")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(project="mgca_finetune", save_dir=logger_dir, name=f"{args.dataset}_{args.data_pct}_{extension}")
    trainer = Trainer.from_argparse_args(args, deterministic=True, callbacks=callbacks, logger=wandb_logger, gpus=1)

    tuner.training_steps = tuner.num_training_steps(trainer, datamodule)
    print(tuner.training_steps)

    # train
    trainer.fit(tuner, datamodule=datamodule)
    # test
    trainer.test(tuner, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()



