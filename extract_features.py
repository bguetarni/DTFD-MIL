import argparse, os, glob, math, random, re
import pandas
import numpy as np
import tqdm
from skimage import filters, morphology
import openslide
from PIL import Image
import torch
import torchvision.transforms as T

from Model.resnet import resnet50_baseline


def scale_coordinates(wsi, p, source_level, target_level):

    if not isinstance(p, np.ndarray):
        p = np.asarray(p).squeeze()

    assert p.ndim < 3 and p.shape[-1] == 2, 'coordinates must be a single point or an array of 2D-cooridnates'

    # source level dimensions
    source_w, source_h = wsi.level_dimensions[source_level]
    
    # target level dimensions
    target_w, target_h = wsi.level_dimensions[target_level]
    
    # scale coordinates
    p = np.array(p)*(target_w/source_w, target_h/source_h)
    
    # round to int64
    return np.floor(p).astype('int64')

def get_tissue_mask_hsv(wsi, mask_level, black_threshold=100):
    """
    Args:
        slide : whole slide file (openslide.OpenSlide)
        mask_level : level from which to build the mask (int)

    Return np.ndarray of bool as mask
    """
    
    # get slide image into PIL.Image
    thumbnail = wsi.read_region((0, 0), mask_level, wsi.level_dimensions[mask_level])

    # convert to HSV
    hsv = np.array(thumbnail.convert('HSV'))
    H, S, V = np.moveaxis(hsv, -1, 0)

    # filter out black pixels
    V_mask = V > black_threshold

    S_filtered = S[V_mask]
    S_threshold = filters.threshold_otsu(S_filtered)
    S_mask = S > S_threshold

    H_filtered = H[V_mask]
    H_threshold = filters.threshold_otsu(H_filtered)
    H_mask = H > H_threshold

    mask = np.logical_and(np.logical_and(H_mask, S_mask), V_mask)
    mask = morphology.binary_dilation(mask)

    return mask

def get_tile_mask(wsi, level, mask, mask_level, x, y, psize):
    # convert coordinates from slide level to mask level
    x_ul, y_ul = scale_coordinates(wsi, (x, y), level, mask_level)
    x_br, y_br = scale_coordinates(wsi, (x + psize, y + psize), level, mask_level)
    return mask[y_ul:y_br, x_ul:x_br]

def extract_wsi_patches(path, args):
    # load slide file
    wsi = openslide.OpenSlide(path)
    
    # create tissue and annotation mask
    mask = get_tissue_mask_hsv(wsi, args.mask_level)
    
    w, h = wsi.level_dimensions[args.level]
    coord = []
    data = []
    for x in range(0, (w - args.psize) + 1, args.psize):
        for y in range(0, (h - args.psize) + 1, args.psize):
            
            tile_mask = get_tile_mask(wsi, args.level, mask, args.mask_level, x, y, args.psize)
            
            # sometimes we step outside the mask due to coordinates scaling
            if tile_mask.size == 0:
                continue
            
            # check mask cover enough patch
            ratio = np.count_nonzero(tile_mask) / tile_mask.size
            if ratio < args.threshold:
                continue

            patch = wsi.read_region((x, y), args.level, (args.psize, args.psize)).convert('RGB')
            data.append(patch)
            coord.append((x, y))
    
    return coord, data

def extract_save_features(extractor, patches, device, args):

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

    if args.dataset == "dlbclmorph":
        load_patch = lambda p : Image.open(p).convert('RGB').reduce(2)
        patches = map(load_patch, patches)
    elif args.dataset == "bci":
        img = Image.open(patches[0]).convert('RGB').reduce(2)
        patches = patch_region(img, psize=256)
    
    patches = list(map(transform, patches))
    features = []
    for batch in batch_list(patches, args.batch_extract):
        batch = torch.stack(batch, dim=0).to(device=device)
        with torch.no_grad():
            fts = extractor(batch)
            features.append(fts)
    
    features = torch.cat(features, dim=0).to(device='cpu')
    return features


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # extractor
    extractor = resnet50_baseline(pretrained=True).to(device=device)
    extractor.eval()

    if args.dataset == "dlbclmorph":
        for fold in os.listdir(args.fold):
            print("fold ", fold)
            for p in tqdm.tqdm(os.listdir(os.path.join(args.fold, fold)), ncols=50):
                for stain in os.listdir(args.src):

                    # get patient patches
                    patches = glob.glob(os.path.join(args.src, stain, p, "*.png"))
                    
                    # extract patches features
                    with torch.no_grad():
                        fts = extract_save_features(extractor, patches, device, args)

                    dst_path = os.path.join(args.dst, fold, p, stain)
                    os.makedirs(dst_path, exist_ok=True)
                    
                    # save patches features
                    for patch_path, f in zip(patches, fts):
                        torch.save(f.clone(), os.path.join(dst_path, "{}.pt".format(os.path.splitext(os.path.split(patch_path)[1])[0])))
    elif args.dataset == "bci":
        for split in os.listdir(args.src):
            print("split ", split)
            images = glob.glob(os.path.join(args.src, split, "*.png"))
            
            # for each image get features
            for img in tqdm.tqdm(images, ncols=50):
                with torch.no_grad():
                    fts = extract_save_features(extractor, [img], device, args)
                
                img_name = os.path.splitext(os.path.split(img)[1])[0]
                dst_path = os.path.join(args.dst, split, img_name)
                os.makedirs(dst_path, exist_ok=True)
                
                # save image patches features
                for i, f in enumerate(fts):
                    torch.save(f.clone(), os.path.join(dst_path, "{}.pt".format(i)))
    elif args.dataset == "chulille":
        labels = pandas.read_csv(args.labels).set_index('patient_id')['label'].to_dict()
        for stain in os.listdir(args.src):
            print(stain)
            for slide in tqdm.tqdm(os.listdir(os.path.join(args.src, stain)), ncols=50):
                p = re.findall('\d+', slide)[0]
                if (not int(p) in labels.keys()) or (not labels[int(p)] in ["ABC", "GCB"]):
                    continue

                slide_dir = os.path.join(args.dst, p, stain, os.path.splitext(slide)[0])
                os.makedirs(slide_dir, exist_ok=True)

                coord, patches = extract_wsi_patches(os.path.join(args.src, stain, slide), args)

                if len(patches) == 0:
                    print("WARNING: slide {} (stain {}) has no patches.".format(slide, stain))
                    continue

                features = extract_save_features(extractor, patches, device, args)

                for xy, fts in zip(coord, features):
                    torch.save(fts.clone(), os.path.join(slide_dir, "{}_{}.pt".format(*xy)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["chulille", "dlbclmorph", "bci"])
    parser.add_argument('--gpu', required=True, type=str)
    parser.add_argument('--src', required=True, type=str)
    parser.add_argument('--dst', required=True, type=str)
    parser.add_argument('--fold', default=None, type=str)
    parser.add_argument('--batch_extract', required=True, type=int)
    parser.add_argument('--labels', type=str, default=None, help='path to labels CSV file')

    parser.add_argument('--level', type=int, required=True, help='WSI level to use')
    parser.add_argument('--mask_level', type=int, required=True, help='level for tissue mask creation')
    parser.add_argument('--psize', type=int, required=True, help='patch size')
    parser.add_argument('--threshold', type=float, required=True, help='threshold for tissue')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.manual_seed(32)
    torch.cuda.manual_seed(32)
    np.random.seed(32)
    random.seed(32)

    main(args)
