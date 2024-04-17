import os
import re
import nltk
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.utils.data as data

from tqdm import tqdm
from clip import clip
from transformers import BertTokenizer
# from constants import *
from mgca.constants import *
from mgca.datasets.utils import get_imgs
from nltk.tokenize import RegexpTokenizer
# from utils import get_imgs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 75 44
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

class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0, imsize=256, max_words=112, sent_num=3):
        super().__init__()
        if not os.path.exists(MIMIC_CXR_DATA_DIR): # ./data/mimic_cxr/
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")

        self.transform = transform
        self.imsize = imsize
        self.df = pd.read_csv(MIMIC_CXR_MASTER_CSV) # master.csv
        self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:]))) # path

        ''''''
        # fill na with 0s
        self.df = self.df.fillna(0)
        # replace uncertains
        # uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
        uncertain_mask = {k: -1 for k in CHEXPERT_TASKS}
        self.df = self.df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)
        ''''''

        # load studies and study to text mapping
        self.filenames, self.fileidxs, self.path2sent = self.load_text_data(split) # 图像名 / 文件夹名

        self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split]    # 提取训练数据
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)
        
        self.tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words

    def load_text_data(self, split):
        # get study to captions mapping
        # TODO: check this
        filepath = os.path.join(BASE_DIR, "captions.pickle")

        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        fileidxs = []
        for row in self.df.itertuples():
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
            path = getattr(row, MIMIC_CXR_PATH_COL)
            ''''''
            idx = []
            for i in range(14):
                idx.append(row[i+9])
            
            ''''''
            if cur_split == split and path in path2sent:
                filenames.append(path)
                fileidxs.append(idx)
        
        return filenames, fileidxs, path2sent

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}
        # iterrows is not faster than itertuples ...  but it is ok
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # pick impression, findings, last_paragraph
            captions = ""
            # captions += row["impression"]
            # captions += " "
            captions += row["findings"]

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())
                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            if cnt >= 3:
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[row[MIMIC_CXR_PATH_COL]] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]")
        print(f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]")

        return path2sent

    def __len__(self):
        return len(self.filenames)

    def get_caption(self, path):
        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents) # 提取的文本 Funding
        
        # 提取关键词
        sent = sent.lower()
        sent_token = nltk.word_tokenize(sent) # 提取token
        # 按列表过滤
        JP_expect = [s1 for s1 in JP_list if s1 in sent_token]
        BL_expect = [s1 for s1 in BL_list if s1 in sent_token]
        
        JP_idx = [pos for pos, char in enumerate(JP_list) if char in JP_expect]
        if JP_idx == []:
            JP_idx = [75]
        JP_labels = sum(F.one_hot(torch.tensor(JP_idx), 76))
        
        BL_idx = [pos for pos, char in enumerate(BL_list) if char in BL_expect]
        if BL_idx == []:
            BL_idx = [44]
        BL_labels = sum(F.one_hot(torch.tensor(BL_idx), 45))
        
        JP_sent = ' '.join(JP_expect)
        BL_sent = ' '.join(BL_expect)
        # JP_full = ' '.join(JP_list)
        # BL_full = ' '.join(BL_list)
        
        token = clip.tokenize(sent, context_length=768)
        tokens = clip.tokenize(sent, truncate=True)      # truncate=True 大于77只取77

        JP_tokens = clip.tokenize(JP_sent, truncate=True)
        BL_tokens = clip.tokenize(BL_sent, truncate=True)
        # JP_fulls = clip.tokenize(JP_full, context_length=160)
        # BL_fulls = clip.tokenize(BL_full, context_length=102)
        
        toke = self.tokenizer(sent, return_tensors="pt", truncation=True, padding="max_length", max_length=77)
        x_len = len([t for t in toke["input_ids"][0] if t != 0])

        return token, tokens, JP_tokens, BL_tokens, JP_labels, BL_labels, x_len

    def __getitem__(self, index):
        key = self.filenames[index]
        idx = self.fileidxs[index]
        caps, caps_77, caps_jp, caps_bl, lbs_jp, lbs_bl, cap_len = self.get_caption(key)
        imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        ''''''
        # row = self.df.iloc[index]
        # gt = list(row[CHEXPERT_COMPETITION_TASKS])
        # gt = list(row[CHEXPERT_TASKS])
        gts = torch.tensor(idx)
        ''''''
        return imgs, caps, caps_77, caps_jp, caps_bl, lbs_jp, lbs_bl, cap_len, key, gts # 图像 文本 文本长度 图像路径


def multimodal_collate_fn(batch):
    """sort sequence"""
    imgs, cap_len, tokens, tokens_77, tokens_JP, tokens_BL = [], [], [], [], [], []
    labels_JP, labels_BL, labels = [], [], []
    path = []
    
    for b in batch:
        img, cap, cap_77, cap_jp, cap_bl, lb_jp, lb_bl, cap_l, p, gt = b
        imgs.append(img)
        tokens.append(cap)
        tokens_77.append(cap_77)
        tokens_JP.append(cap_jp)
        tokens_BL.append(cap_bl)
        labels_JP.append(lb_jp)
        labels_BL.append(lb_bl)
        labels.append(gt)
        cap_len.append(cap_l)
        path.append(p)

    # stack
    imgs = torch.stack(imgs)
    tokens = torch.stack(tokens).squeeze()
    tokens_77 = torch.stack(tokens_77).squeeze()
    tokens_JP = torch.stack(tokens_JP).squeeze()
    tokens_BL = torch.stack(tokens_BL).squeeze()
    labels_JP = torch.stack(labels_JP).squeeze()
    labels_BL = torch.stack(labels_BL).squeeze()
    labels = torch.stack(labels)
    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    path = np.array(path)

    return_dict = {
        "cap_lens": sorted_cap_lens,
        "imgs": imgs[sorted_cap_indices],
        "path": path[sorted_cap_indices],
        "gts": labels[sorted_cap_indices],
        "tokens": tokens[sorted_cap_indices],
        "tokens_77": tokens_77[sorted_cap_indices],
        "tokens_JP": tokens_JP[sorted_cap_indices], 
        "tokens_BL": tokens_BL[sorted_cap_indices], 
        "labels_JP": labels_JP[sorted_cap_indices], 
        "labels_BL": labels_BL[sorted_cap_indices] }
    return return_dict


if __name__ == "__main__":
    from mgca.datasets.transforms import DataTransforms
    # from transforms import DataTransforms
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="train", transform=transform)
    data = dataset[0]
    print(dataset)




