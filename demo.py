import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from raft import RAFT
from utils import flow_viz, frame_utils
from utils.utils import InputPadder
from pathlib import Path
from ipdb import set_trace
from natsort import natsorted



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()

def load_model(args) :
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    return model

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = natsorted(images)
        for imfile1, imfile2 in tqdm(zip(images[:-1], images[1:]), total=len(images)):
            try :
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True) # Flow Up is the upsampled version

            if args.save :
                path = Path(args.path_save)
                path.mkdir(parents=True, exist_ok=True)
                flow = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()
                frame_utils.writeFlow(imfile1.replace(args.path,args.path_save).replace('.png','.flo'), flow)
            else :
                viz(image1, flow_up)
                
            except Exception as e :
                print(f'Error with {imfile1} : {e}')

@torch.no_grad()
def compute_flow_dir(model, dirpath, dirpathsave, resize=None) :
    images = glob.glob(os.path.join(dirpath, '*.png')) + \
                 glob.glob(os.path.join(dirpath, '*.jpg'))

    images = natsorted(images)
    for imfile1, imfile2 in tqdm(zip(images[:-1], images[1:]), total=len(images)):
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)
        extension=imfile1.split('.')[-1]

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True) # Flow Up is the upsampled version
        if resize is not None :
            flow_up = nn.functional.interpolate(flow_up, size=resize, mode='bilinear', align_corners=False)


        path = Path(dirpathsave)
        path.mkdir(parents=True, exist_ok=True)
        flow = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()
        frame_utils.writeFlow(imfile1.replace(dirpath, dirpathsave).replace(extension,'flo'), flow)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--save', action='store_true', help='save the frame instead of showing them')
    parser.add_argument('--path_save', required='--save' in sys.argv, type=str)

    args = parser.parse_args()

    demo(args)
