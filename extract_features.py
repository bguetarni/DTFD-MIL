import argparse
import json
import os
import pandas
import glob
import math
import pickle
import random
import numpy as np
import tqdm
from PIL import Image

import torch.nn.functional as F
import torch
import torchvision.transforms as T
from Model.resnet import resnet50_baseline


def extract_save_features(extractor, patches, device, params):

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    patches = list(map(transform, map(Image.open, patches)))
    features = []
    for batch in np.array_split(patches, math.ceil(len(patches) / params.batch_extract)):
        batch = torch.stack(list(batch), dim=0).to(device=device)
        fts = extractor(batch)
        features.append(fts)
    
    features = torch.cat(features, dim=0)
    return features


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # extractor
    extractor = resnet50_baseline(pretrained=True).to(device=device)
    for p in extractor.parameters():
        p.requires_grad = False
    extractor.eval()

    for p in tqdm.tqdm(os.listdir(params.src), ncols=100):
        for s in os.listdir(os.path.join(params.src, p)):
            patches = glob.glob(os.path.join(params.src, p, s, "*.png"))

            with torch.no_grad():
                fts = extract_save_features(extractor, patches, device, params)

            for patch_path, f in zip(patches, fts):
                dst_path = os.path.join(params.dst, p, s, "{}.pt".format(os.path.splitext(os.path.split(patch_path)[1])[0]))
                if not os.path.isdir(os.path.split(dst_path)[0]):
                    os.makedirs(os.path.split(dst_path)[0])

                torch.save(f.detach().to('cpu'), dst_path)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', required=True, type=str)
parser.add_argument('--src', required=True, type=str)
parser.add_argument('--dst', required=True, type=str)
parser.add_argument('--batch_extract', required=True, type=int)
params = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

torch.manual_seed(32)
torch.cuda.manual_seed(32)
np.random.seed(32)
random.seed(32)

if __name__ == "__main__":
    main(params)