import argparse, os, glob, math, random
import numpy as np
import tqdm
from PIL import Image
import torch
import torchvision.transforms as T

from Model.resnet import resnet50_baseline


def extract_save_features(extractor, patches, device, params):

    def patch_region(img, psize):
        img = np.array(img)
        s = img.shape[0]
        splitted = np.stack(np.split(img, s//psize, axis=1), axis=0)
        splitted = np.stack(np.split(splitted, s//psize, axis=1), axis=0)
        splitted = splitted.reshape((-1, *splitted.shape[2:]))
        return splitted
    
    def batch_list(data, batch_size):
        idx = range(0, len(data), batch_size)
        return [data[i:i+batch_size] for i in idx]

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if params.dataset in ["chulille", "dlbclmorph"]:
        load_patch = lambda p : Image.open(p).convert('RGB').reduce(2)
        patches = map(load_patch, patches)
    elif params.dataset == "bci":
        img = Image.open(patches[0]).convert('RGB').reduce(2)
        patches = patch_region(img, psize=256)
    
    patches = list(map(transform, patches))
    features = []
    for batch in batch_list(patches, params.batch_extract):
        batch = torch.stack(batch, dim=0).to(device=device)
        with torch.no_grad():
            fts = extractor(batch)
            features.append(fts)
    
    features = torch.cat(features, dim=0)
    return features


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # extractor
    extractor = resnet50_baseline(pretrained=True).to(device=device)
    extractor.eval()

    if params.dataset == "dlbclmorph":
        for fold in os.listdir(params.fold):
            print("fold ", fold)
            for p in tqdm.tqdm(os.listdir(os.path.join(params.fold, fold)), ncols=50):
                for stain in os.listdir(params.src):

                    # get patient patches
                    patches = glob.glob(os.path.join(params.src, stain, p, "*.png"))
                    
                    # extract patches features
                    with torch.no_grad():
                        fts = extract_save_features(extractor, patches, device, params)

                    dst_path = os.path.join(params.dst, fold, p, stain)
                    os.makedirs(dst_path, exist_ok=True)
                    
                    # save patches features
                    for patch_path, f in zip(patches, fts):
                        torch.save(f.detach().to('cpu'), os.path.join(dst_path, "{}.pt".format(os.path.splitext(os.path.split(patch_path)[1])[0])))
    elif params.dataset == "bci":
        for split in os.listdir(params.src):
            print("split ", split)
            images = glob.glob(os.path.join(params.src, split, "*.png"))
            
            # for each image get features
            for img in tqdm.tqdm(images, ncols=50):
                with torch.no_grad():
                    fts = extract_save_features(extractor, [img], device, params)
                
                img_name = os.path.splitext(os.path.split(img)[1])[0]
                dst_path = os.path.join(params.dst, split, img_name)
                os.makedirs(dst_path, exist_ok=True)
                
                # save image patches features
                for i, f in enumerate(fts):
                    torch.save(f.detach().to('cpu'), os.path.join(dst_path, "{}.pt".format(i)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["chulille", "dlbclmorph", "bci"])
    parser.add_argument('--gpu', required=True, type=str)
    parser.add_argument('--src', required=True, type=str)
    parser.add_argument('--dst', required=True, type=str)
    parser.add_argument('--fold', default=None, type=str)
    parser.add_argument('--batch_extract', required=True, type=int)
    params = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

    torch.manual_seed(32)
    torch.cuda.manual_seed(32)
    np.random.seed(32)
    random.seed(32)

    main(params)
