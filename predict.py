import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from config import model_name
from utils.data_loading import BasicDataset
from unet import UNetExtractor_v2, MSEA_unet_v3, MSEA_unet_v2, UNet

from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images or a directory of images')
    parser.add_argument('--model', '-m', default=r"D:\Thesis\Pytorch-UNet-master\Pytorch-UNet-master\checkpoints\MSEA_unet_v3_20240822_150500_20.pth", metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', default=r"D:/Thesis/datasets/main_DATASET/AUGMENTED/Test/img/", nargs='+', help='Filenames or directory of input images')
    parser.add_argument('--output-dir', '-o', metavar='OUTPUT_DIR', default=r'./results/new 80/', help='Directory for output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    
    return parser.parse_args()

def get_output_filenames(args, in_files):
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return [os.path.join(output_dir, os.path.basename(fn)) for fn in in_files]

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Check if input is a directory and get all images from it
    in_files = []
    if os.path.isdir(args.input[0]):
        for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'):
            in_files.extend([os.path.join(args.input[0], fn) for fn in os.listdir(args.input[0]) if fn.lower().endswith(ext)])
    else:
        in_files = args.input

    out_files = get_output_filenames(args, in_files)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    net = globals()[model_name](n_channels=3, dim=32, n_classes=args.classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    checkpoint = torch.load(args.model, map_location=device)
    mask_values = checkpoint.pop('mask_values', [0, 1])
    net.load_state_dict(checkpoint)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
