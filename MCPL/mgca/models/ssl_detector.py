import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from PIL import Image
from PIL import ImageDraw
from pprint import pprint
from typing import List
from collections import OrderedDict
from pytorch_lightning import LightningModule

from mgca.utils.yolo_loss import YOLOLoss
from mgca.datasets.data_module import DataModule
from mgca.datasets.transforms import DataTransforms
from mgca.utils.detection_utils import non_max_suppression
from mgca.datasets.detection_dataset import RSNADetectionDataset

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

class SSLDetector(LightningModule):
    def __init__(self, img_encoder: nn.Module, learning_rate: float=5e-4, weight_decay: float=1e-6, imsize: int=224,
                 conf_thres: float=0.5, iou_thres: List = [0.5], nms_thres: float=0.4, *args, **kwargs):
        super().__init__()

        self.model = ModelMain(img_encoder) # YOLO 
        self.yolo_losses = []

        for i in range(3):
            self.yolo_losses.append(YOLOLoss(self.model.anchors[i], self.model.classes, (imsize, imsize)))

        self.iou_thres = iou_thres
        self.nms_thres = nms_thres
        self.conf_thres = conf_thres
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        # self.train_map = MeanAveragePrecision(box_format='cxcywh')
        self.obj_map = MeanAveragePrecision(iou_thresholds=self.iou_thres, box_format='xyxy')
        self.val_map = MeanAveragePrecision(iou_thresholds=self.iou_thres, box_format='xyxy')
        self.test_map = MeanAveragePrecision(iou_thresholds=self.iou_thres, box_format='xyxy')
        
    def shared_step(self, batch, batch_idx, split):
        outputs = self.model(batch["imgs"])

        # ''''''
        # A = F.sigmoid(torch.mean(outputs[2], dim=1)) # 18 28 28
        # B = batch["imgs"]
        # ''''''

        losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
        losses = []
        for _ in range(len(losses_name)):
            losses.append([])

        for i in range(3):
            _loss_item = self.yolo_losses[i](outputs[i], batch["labels"])
            for j, l in enumerate(_loss_item):
                losses[j].append(l)
        losses = [sum(l) for l in losses]
        loss = losses[0]

        self.log(f"{split}_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        if split != "train":
            output_list = []
            
            for i in range(3): ###### 
                output_list.append(self.yolo_losses[i](outputs[i]))

            output = torch.cat(output_list, 1)
            output = non_max_suppression(output, self.model.classes, conf_thres=self.conf_thres, nms_thres=self.nms_thres)
            targets = batch["labels"].clone()

            # cxcywh -> xyxy
            h, w = batch["imgs"].shape[2:]
            targets[:, :, 1] = (batch["labels"][..., 1] - batch["labels"][..., 3]/2) * 224
            targets[:, :, 2] = (batch["labels"][..., 2] - batch["labels"][..., 4]/2) * 224
            targets[:, :, 3] = (batch["labels"][..., 1] + batch["labels"][..., 3]/2) * 224
            targets[:, :, 4] = (batch["labels"][..., 2] + batch["labels"][..., 4]/2) * 224

            sample_preds, sample_targets = [], []
            IOU, MAP = [], []

            for i in range(targets.shape[0]):
                out = output[i]
                target = targets[i]
                filtered_target = target[target[:, 1] > 0.0]
                
                if out is None:
                    continue

                if filtered_target.shape[0] > 0:
                    target_box = filtered_target[:, 1:5]

                    # ''''''
                    # AM = F.upsample(A[i:i+1].reshape(1,1,28,28).data, size=(224, 224), mode='bilinear')
                    # BM = B[i:i+1]
                    # CM = Image.fromarray(np.zeros((224, 224, 3)).astype(np.uint8))
                    # # BM = B[i].permute(1, 2, 0).cpu().numpy()
                    # # BM = Image.fromarray((255*BM).astype(np.uint8))
                    # CD = target_box[0,:].cpu().numpy()
                    # draw = ImageDraw.Draw(CM)
                    # draw.rectangle([round(CD[0]), round(CD[1]), round(CD[2]), round(CD[3])], outline=(255,0,0)) # [左上角x，左上角y，右下角x，右下角y], outline边框颜色
                    # # CM.show()
                    # # BM = torch.from_numpy(np.array(BM)/255).permute(2, 0, 1).view(1, 3, 224, 224)
                    # CM = torch.from_numpy(np.array(CM)/255).permute(2, 0, 1).view(1, 3, 224, 224)

                    # vutils.save_image(AM, './vision2/AM'+str(batch_idx)+'_'+str(i)+'.png', normalize=False, scale_each=False)
                    # vutils.save_image(BM, './vision2/BM'+str(batch_idx)+'_'+str(i)+'.png', normalize=True, scale_each=False)
                    # vutils.save_image(CM, './vision2/CM'+str(batch_idx)+'_'+str(i)+'.png', normalize=False, scale_each=False)
                    # ''''''

                    m_iou = 0.0
                    predict_box = out[:, 0:4]

                    for j in range(filtered_target.shape[0]):
                        iou = bbox_iou(predict_box, target_box[j:j+1, :], x1y1x2y2=True)
                        m_iou += torch.mean(iou)
                    m_iou = m_iou/filtered_target.shape[0]
                    IOU.append(m_iou)

                    # 每张图像的 predict 和 target 之间的map
                    sample_tar = dict(boxes=filtered_target[:, 1:5]/224, labels=filtered_target[:, 0])
                    sample_targets.append(sample_tar)
                    sample_pre = dict(boxes=out[:, 0:4]/224, scores=out[:, 4], labels=out[:, 6])
                    sample_preds.append(sample_pre)
                    
                    self.obj_map.update([sample_pre], [sample_tar])
                    m_map = self.obj_map.compute()["map"]
                    self.obj_map.reset()
                    MAP.append(m_map)

            if split == "valid":
                if len(IOU) == 0 and len(MAP) == 0:
                    IOU.append(torch.tensor(0.0))
                    MAP.append(torch.tensor(0.0))
                self.V_IOU = torch.mean(torch.stack(IOU))
                self.V_MAP = torch.mean(torch.stack(MAP))
                self.val_map.update(sample_preds, sample_targets)
            elif split == "test":
                self.T_IOU = torch.mean(torch.stack(IOU))
                self.T_MAP = torch.mean(torch.stack(MAP))
                self.test_map.update(sample_preds, sample_targets)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "train")
        if batch_idx % 40 == 0:
            print('Train Batch: [%d] | Loss: %.4f' % (batch_idx, loss) )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "valid")
        if batch_idx % 10 == 0:
            print('Val Batch: [%d] | Loss: %.4f | Val_IOU: %.4f | Val_MAP: %.4f' % (batch_idx, loss, self.V_IOU, self.V_MAP) )
        return loss

    def test_step(self, batch, batch_idx):
        print('Test Batch: [%d] | Test_IOU: %.4f | Test_MAP: %.4f' % (batch_idx, self.T_IOU, self.T_MAP) )
        return self.shared_step(batch, batch_idx, "test")

    def validation_epoch_end(self, validation_step_outputs):
        map = self.val_map.compute()["map"]
        self.log("val_mAP", map, prog_bar=True, on_epoch=True, sync_dist=True)
        print('V_MAP: %.4f' % (map) )
        self.val_map.reset()

    def test_epoch_end(self, test_step_outputs):
        map = self.test_map.compute()["map"]
        self.log("test_mAP", map, prog_bar=True, on_epoch=True, sync_dist=True)
        print('T_MAP: %.4f' % (map) )
        self.test_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        # return optimizer
        
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=self.training_steps, cycle_mult=1.0,
                                                     max_lr=self.learning_rate, min_lr=1e-6, warmup_steps=int(self.training_steps * 0.2))
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * trainer.max_epochs


class ModelMain(nn.Module):
    def __init__(self, backbone, is_training=True):
        super(ModelMain, self).__init__()
        self.training = is_training
        self.backbone = backbone
        self.anchors = torch.tensor([ [[116, 90], [156, 198], [373, 326]],
                                      [[30, 61], [62, 45], [59, 119]],
                                      [[10, 13], [16, 30], [33, 23]] ]) * 224 / 416
        self.classes = 1

        _out_filters = self.backbone.filters
        #  embedding0
        final_out_filter0 = len(self.anchors[0]) * (5 + self.classes)
        self.embedding0 = self._make_embedding([512, 1024], _out_filters[-1], final_out_filter0)
        #  embedding1
        final_out_filter1 = len(self.anchors[1]) * (5 + self.classes)
        self.embedding1_cbl = self._make_cbl(512, 256, 1)
        self.embedding1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding1 = self._make_embedding([256, 512], _out_filters[-2] + 256, final_out_filter1)
        #  embedding2
        final_out_filter2 = len(self.anchors[2]) * (5 + self.classes)
        self.embedding2_cbl = self._make_cbl(256, 128, 1)
        self.embedding2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding2 = self._make_embedding([128, 256], _out_filters[-3] + 128, final_out_filter2)

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu'''
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([ ("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False)),
                                           ("bn", nn.BatchNorm2d(_out)),
                                           ("relu", nn.LeakyReLU(0.1)) ]))

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3) ])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True))
        return m

    def forward(self, x):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch

        #  backbone
        # x2: bz, 512, 28, 28
        # x1: bz, 1024, 14, 14
        # x0: bz, 2048, 7, 7
        x2, x1, x0 = self.backbone(x)

        #  yolo branch 0
        out0, out0_branch = _branch(self.embedding0, x0)
        #  yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        x1_in = self.embedding1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)
        #  yolo branch 2
        x2_in = self.embedding2_cbl(out1_branch)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)

        # out0: bz, 18, 7, 7
        # out1: bz, 18, 14, 14
        # out2: bz, 18, 28, 28

        return out0, out1, out2


if __name__ == "__main__":
    model = ModelMain()

    datamodule = DataModule(RSNADetectionDataset, None, DataTransforms, 0.1, 32, 1, 224)

    for batch in datamodule.train_dataloader():
        break




