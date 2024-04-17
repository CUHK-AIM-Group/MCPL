
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from clip import clip
from torch.autograd import Variable

import warnings
warnings.filterwarnings("ignore") 



import open_clip

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

# from huggingface_hub import snapshot_download
# snapshot_download(repo_id='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir='D:\BiomedCLIP')


import glob
from collections import OrderedDict

import torch
from PIL import Image

dataset_path = './example_data/biomed_image_classification_example_data'
template = 'this is a photo of '
labels = [
    'adenocarcinoma histopathology',
    'brain MRI',
    'covid line chart',
    'squamous cell carcinoma histopathology',
    'immunohistochemistry histopathology',
    'bone X-ray',
    'chest X-ray',
    'pie chart',
    'hematoxylin and eosin histopathology'
]

test_imgs = glob.glob(dataset_path + '/*')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

context_length = 256

images = torch.stack([preprocess_val(Image.open(img)) for img in test_imgs]).to(device)
texts = tokenizer([template + l for l in labels], context_length=context_length).to(device)
with torch.no_grad():
    image_features, text_features, logit_scale = model(images, texts)

    logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)

    logits = logits.cpu().numpy()
    sorted_indices = sorted_indices.cpu().numpy()

top_k = -1

for i, img in enumerate(test_imgs):
    pred = labels[sorted_indices[i][0]]

    top_k = len(labels) if top_k == -1 else top_k
    print(img.split('/')[-1] + ':')
    for j in range(top_k):
        jth_index = sorted_indices[i][j]
        print(f'{labels[jth_index]}: {logits[i][jth_index]}')
    print('\n')
    

# JP_list = ['pleural', 'lung', 'pulmonary', 'lungs', 'chest', 'silhouette', 'mediastinal', 
#             'cardiomediastinal', 'heart', 'hilar', 'tube', 'osseous', 'lobe', 'vascular', 
#             'thoracic', 'catheter', 'interval', 'bibasilar', 'aorta', 'vasculature', 'interstitial', 
#             'svc', 'spine', 'silhouettes', 'rib', 'otracheal', 'bony', 'sternotomy', 
#             'stomach', 'retrocardiac', 'aortic', 'basilar', 'picc', 'clips', 'costophrenic', 
#             'abdomen', 'atrium', 'wires', 'venous', 'nasogastric', 'fluid', 'ventricle', 
#             'pacemaker', 'jugular', 'bronchovascular', 'vascularity', 'enteric', 'hila', 'diaphragm', 
#             'perihilar', 'port-a-cath', 'arch', 'hemithorax', 'subclavian', 'tissue', 'cavoatrial', 
#             'knob', 'vertebral', 'tracheostomy', 'valve', 'pacer', 'artery', 'hiatal', 
#             'trachea', 'vein', 'cabg', 'subcutaneous', 'tubes', 'esophagus', 'stent', 
#             'vessels', 'cervical', 'sternal', 'neck', 'junction']

# JP_list = ['lungs', 'valve', 'cervical']

# JP_sent = ' '.join(JP_list)
# JP_tokens = clip.tokenize(JP_sent)   
# JP_mask = torch.masked_select(JP_tokens, JP_tokens>1)
# # token_embed = clip.token_embedding        
# print(JP_tokens>1)

# from class_2 import ClassFineTuner
# model = ClassFineTuner.load_from_checkpoint('1.ckpt', strict=False)
# # IE = model.image_encoder
# # TE = model.text_encoder
# # PL = model.prompt_learner.ctx_text
# RS = model.backbone
# print(RS)
# torch.save(RS, 'RS1.pth')

# def bbox_iou(box1, box2, x1y1x2y2=True):
#     """
#     Returns the IoU of two bounding boxes
#     """
#     if not x1y1x2y2:
#         # Transform from center and width to exact coordinates
#         b1_x1, b1_x2 = box1[:, 0] - box1[:, 2]/2, box1[:, 0] + box1[:, 2]/2
#         b1_y1, b1_y2 = box1[:, 1] - box1[:, 3]/2, box1[:, 1] + box1[:, 3]/2
#         b2_x1, b2_x2 = box2[:, 0] - box2[:, 2]/2, box2[:, 0] + box2[:, 2]/2
#         b2_y1, b2_y2 = box2[:, 1] - box2[:, 3]/2, box2[:, 1] + box2[:, 3]/2
#     else:
#         # Get the coordinates of bounding boxes
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

#     # get the corrdinates of the intersection rectangle
#     inter_rect_x1 = torch.max(b1_x1, b2_x1)
#     inter_rect_y1 = torch.max(b1_y1, b2_y1)
#     inter_rect_x2 = torch.min(b1_x2, b2_x2)
#     inter_rect_y2 = torch.min(b1_y2, b2_y2)
#     # Intersection area
#     inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
#         torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
#     # Union Area
#     b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
#     b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

#     iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
#     return iou

# from torch import tensor
# box1=tensor([[10, 10, 30, 20],
#              [30, 10, 50, 20],
#              ])
# box2=tensor([[30, 10, 50, 20]])

# iou = bbox_iou(box1, box2, x1y1x2y2=True)
# print(iou)


# from pprint import pprint
# from torchmetrics.detection import MeanAveragePrecision

# # preds = [dict(boxes=tensor([[10, 10, 30, 20],
# #                             [10, 10, 30, 20],
# #                             [10, 10, 30, 20],
# #                             [30, 10, 50, 20],
# #                             ]),
# #               scores=tensor([1.0, 1.0, 1.0, 0.5]),
# #               labels=tensor([0, 0, 0, 0]) )]

# preds = [dict(boxes=tensor([[0.49, 0.39, 0.69, 0.49],
#                             [0.29, 0.09, 0.49, 0.19],
#                             ]),
#               scores=tensor([0.1, 0.1]),
#               labels=tensor([0, 0]) )]

# target = [dict(boxes=tensor([[0.30, 0.10, 0.50, 0.20],
#                              [0.40, 0.20, 0.60, 0.30]
#                              ]),
#                 labels=tensor([0, 0]) )]

# metric = MeanAveragePrecision(iou_thresholds=[0.5], box_format='xyxy')
# metric.update(preds, target)
# pprint(metric.compute()["map"])
a = 0.5
if a < 1.0 and a > 0.0:
    print(a)


# import pickle

# F=open(r'D:\124\MGCA-main\mgca\data\object-CXR\test.pkl','rb')

# content=pickle.load(F)
# print(content)

# import torch
# from collections import Counter
 
# def mean_average_precision(pred_bboxes,true_boxes,iou_threshold,num_classes=20):
    
#     #pred_bboxes(list): [[train_idx,class_pred,prob_score,x1,y1,x2,y2], ...]
    
#     average_precisions=[]#存储每一个类别的AP
#     epsilon=1e-6#防止分母为0
    
#     #对于每一个类别
#     for c in range(num_classes):
#         detections=[]#存储预测为该类别的bbox
#         ground_truths=[]#存储本身就是该类别的bbox(GT)
        
#         for detection in pred_bboxes:
#             if detection[1]==c:
#                 detections.append(detection)
        
#         for true_box in true_boxes:
#             if true_box[1]==c:
#                 ground_truths.append(true_box)
                
#         #img 0 has 3 bboxes
#         #img 1 has 5 bboxes
#         #就像这样：amount_bboxes={0:3,1:5}
#         #统计每一张图片中真实框的个数,train_idx指示了图片的编号以区分每张图片
#         amount_bboxes=Counter(gt[0] for gt in ground_truths)
        
#         for key,val in amount_bboxes.items():
#             amount_bboxes[key]=torch.zeros(val)#置0，表示这些真实框初始时都没有与任何预测框匹配
#         #此时，amount_bboxes={0:torch.tensor([0,0,0]),1:torch.tensor([0,0,0,0,0])}
        
#         #将预测框按照置信度从大到小排序
#         detections.sort(key=lambda x:x[2],reverse=True)
        
#         #初始化TP,FP
#         TP=torch.zeros(len(detections))
#         FP=torch.zeros(len(detections))
        
#         #TP+FN就是当前类别GT框的总数，是固定的
#         total_true_bboxes=len(ground_truths)
        
#         #如果当前类别一个GT框都没有，那么直接跳过即可
#         if total_true_bboxes == 0:
#             continue
        
#         #对于每个预测框，先找到它所在图片中的所有真实框，然后计算预测框与每一个真实框之间的IoU，大于IoU阈值且该真实框没有与其他预测框匹配，则置该预测框的预测结果为TP，否则为FP
#         for detection_idx,detection in enumerate(detections):
#             #在计算IoU时，只能是同一张图片内的框做，不同图片之间不能做
#             #图片的编号存在第0个维度
#             #于是下面这句代码的作用是：找到当前预测框detection所在图片中的所有真实框，用于计算IoU
#             ground_truth_img=[bbox for bbox in ground_truths if bbox[0]==detections[0]]
            
#             num_gts=len(ground_truth_img)
            
#             best_iou=0
#             for idx,gt in enumerate(ground_truth_img):
#                 #计算当前预测框detection与它所在图片内的每一个真实框的IoU
#                 iou=insert_over_union(torch.tensor(detection[3:]),torch.tensor(gt[3:]))
#                 if iou >best_iou:
#                     best_iou=iou
#                     best_gt_idx=idx
#             if best_iou>iou_threshold:
#                 #这里的detection[0]是amount_bboxes的一个key，表示图片的编号，best_gt_idx是该key对应的value中真实框的下标
#                 if amount_bboxes[detection[0]][best_gt_idx]==0:#只有没被占用的真实框才能用，0表示未被占用（占用：该真实框与某预测框匹配【两者IoU大于设定的IoU阈值】）
#                     TP[detection_idx]=1#该预测框为TP
#                     amount_bboxes[detection[0]][best_gt_idx]=1#将该真实框标记为已经用过了，不能再用于其他预测框。因为一个预测框最多只能对应一个真实框（最多：IoU小于IoU阈值时，预测框没有对应的真实框)
#                 else:
#                     FP[detection_idx]=1#虽然该预测框与真实框中的一个框之间的IoU大于IoU阈值，但是这个真实框已经与其他预测框匹配，因此该预测框为FP
#             else:
#                 FP[detection_idx]=1#该预测框与真实框中的每一个框之间的IoU都小于IoU阈值，因此该预测框直接为FP
                
#         TP_cumsum=torch.cumsum(TP,dim=0)
#         FP_cumsum=torch.cumsum(FP,dim=0)
        
#         #套公式
#         recalls = TP_cumsum/(total_true_bboxes+epsilon)
#         precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        
#         #把[0,1]这个点加入其中
#         precisions=torch.cat((torch.tensor([1]), precisions))
#         recalls=torch.cat((torch.tensor([0]),recalls))
#         #使用trapz计算AP
#         average_precisions.append(torch.trapz(precisions,recalls))
#         print(average_precisions)
#         return sum(average_precisions)
#     # return sum(average_precisions)/len(average_precisions) 
 
 
# def insert_over_union(boxes_preds,boxes_labels):
    
#     box1_x1=boxes_preds[...,0:1]
#     box1_y1=boxes_preds[...,1:2]
#     box1_x2=boxes_preds[...,2:3]
#     box1_y2=boxes_preds[...,3:4]#shape:[N,1]
    
#     box2_x1=boxes_labels[...,0:1]
#     box2_y1=boxes_labels[...,1:2]
#     box2_x2=boxes_labels[...,2:3]
#     box2_y2=boxes_labels[...,3:4]
    
#     x1=torch.max(box1_x1,box2_x1)
#     y1=torch.max(box1_y1,box2_y1)
#     x2=torch.min(box1_x2,box2_x2)
#     y2=torch.min(box1_y2,box2_y2)
    
    
#     #计算交集区域面积
#     intersection=(x2-x1).clamp(0)*(y2-y1).clamp(0)
    
#     box1_area=abs((box1_x2-box1_x1)*(box1_y1-box1_y2))
#     box2_area=abs((box2_x2-box2_x1)*(box2_y1-box2_y2))
    
#     return intersection/(box1_area+box2_area-intersection+1e-6)








