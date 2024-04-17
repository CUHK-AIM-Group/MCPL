import os
import datetime
from argparse import ArgumentParser

import torch
import torch.nn as nn
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)
# from pytorch_lightning.loggers import WandbLogger

from mgca.datasets.classification_dataset import (MIMICImageDataset, CheXpertImageDataset, COVIDXImageDataset, RSNAImageDataset)
from mgca.datasets.data_module import DataModule
from mgca.datasets.transforms import DataTransforms, Moco2Transform
from mgca.models.ssl_finetuner import SSLFineTuner

# 读取模型
import timm

import open_clip
import clip_class
import torchvision.models as models

from mgca.mm import MGCA
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel


torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rsna")
    parser.add_argument("--name", type=str, default="convnext")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_pct", type=float, default=1.0)
    # add trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # set max epochs
    args.max_epochs = 20
    seed_everything(args.seed)

    if args.dataset == "mimiccxr":
        datamodule = DataModule(MIMICImageDataset, None, DataTransforms,
                                args.data_pct, args.batch_size, args.num_workers)
        num_classes = 14
        multilabel = True
    elif args.dataset == "chexpert":
        # define datamodule
        # check transform here
        datamodule = DataModule(CheXpertImageDataset, None, Moco2Transform,
                                args.data_pct, args.batch_size, args.num_workers)
        num_classes = 14
        multilabel = True
    elif args.dataset == "rsna":
        datamodule = DataModule(RSNAImageDataset, None, DataTransforms,
                                args.data_pct, args.batch_size, args.num_workers)
        num_classes = 2
        multilabel = False
    elif args.dataset == "covidx":
        datamodule = DataModule(COVIDXImageDataset, None, DataTransforms,
                                args.data_pct, args.batch_size, args.num_workers)
        num_classes = 2
        multilabel = False
    else:
        raise RuntimeError(f"no dataset called {args.dataset}")

    if args.name == "clip":
        model, _ = clip_class.load("ViT-B/32", device="cuda") # CLIP
        args.backbone = model.encode_image
        args.in_features = 512
    elif args.name == "medclip":
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT) # MedCLIP
        model.from_pretrained()
        args.backbone = model.vision_model.cuda()
        args.in_features = 512
    elif args.name == "biomedclip":
        model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        args.backbone = model.visual.trunk.cuda().eval()
        args.in_features = 768
    elif args.name == "mgca":
        model = MGCA.load_from_checkpoint("./model/vit_base.ckpt", strict=False)
        args.backbone = model.img_encoder_q.cuda()
        args.in_features = args.backbone.feature_dim
    elif args.name == "imagenet":
        model = models.resnet50(pretrained=True)
        args.backbone = nn.Sequential(*list(model.children())[:-1]).cuda().eval()
        args.in_features = model.fc.in_features
    elif args.name == "vit":
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        args.in_features = model.head.in_features
        model.head = torch.nn.Identity()
        args.backbone = model.cuda().eval()
    elif args.name == "convnext":
        model = timm.create_model('convnext_base', pretrained=True)
        args.in_features = model.head.fc.in_features
        model.head.fc = torch.nn.Identity()
        args.backbone = model.cuda().eval()

    # 共享
    args.num_classes = num_classes
    args.multilabel = multilabel

    # finetune
    tuner = SSLFineTuner(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(BASE_DIR, f"D:/123/MGCA-main/class/cp/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [LearningRateMonitor(logging_interval="step"),
                 ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir, save_last=True, mode="min", save_top_k=1),
                 EarlyStopping(monitor="val_loss", min_delta=0., patience=50, verbose=False, mode="min") ]

    # get current time
    now = datetime.datetime.now(tz.tzlocal())

    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    # logger_dir = os.path.join(BASE_DIR, f"D:/123/MGCA-main/class")
    # os.makedirs(logger_dir, exist_ok=True)
    # wandb_logger = WandbLogger(project="class_cp", save_dir=logger_dir, name=f"{args.dataset}_{args.data_pct}_{extension}")
    # trainer = Trainer.from_argparse_args(args, deterministic=True, callbacks=callbacks, logger=wandb_logger, gpus=1)
    trainer = Trainer.from_argparse_args(args, deterministic=True, callbacks=callbacks, gpus=1)

    tuner.training_steps = tuner.num_training_steps(trainer, datamodule)

    trainer.fit(tuner, datamodule=datamodule) # train
    trainer.test(tuner, datamodule=datamodule, ckpt_path="best") # test


if __name__ == "__main__":
    cli_main()


