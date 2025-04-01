# Originally by SAM J, Authorized by SAM J
# designed for evaluating UCM_LandUse dataset, modified as you need to fit for other dataset

import argparse
import os

import torch
import torchvision.transforms as transform
from torchvision.transforms import InterpolationMode
import cv2
from PIL import Image

import models
from utils_image import calculate_psnr,calculate_ssim

from collections import defaultdict
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model', default='/home/sam/SuperRes/FunSR/checkpoints/Swin_EESCitres_bestval/epoch-best.pth')
    parser.add_argument('--model_name', default='swin_EESCit')
    parser.add_argument('--scale_max', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=48)
    parser.add_argument('--fast', default=False)
    parser.add_argument('--gpu', default='0')
    parser.add_argument("--meta_info", default='/home/sam/SuperRes/EESCit/data/meta_info_UCMx2_testpair.txt', help="dataset info file")
    parser.add_argument(
        "--save_path",
        default='/home/sam/SuperRes/EESCit/test_UCM',
        type=str,
        help="path to store images and if not given, will not save image",
    )
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    trans = transform.Compose([transform.ToTensor(), ])
    img_sub = torch.FloatTensor([0.5]).view(1, -1, 1, 1).to('cuda:0')
    img_div = torch.FloatTensor([0.5]).view(1, -1, 1, 1).to('cuda:0')

    performance = defaultdict(dict)
    performance['PSNR'] = {}
    performance['SSIM'] = {}

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    model.eval()

    hr_dirs = []
    lr_dirs = []
    with open(args.meta_info, 'r') as f:
        for line in f.readlines():
            lr_dir, hr_dir = line.strip().split(',')
            lr_dirs.append(lr_dir)
            hr_dirs.append(hr_dir)

    scale = args.scale_max
    for lr_dir, hr_dir in zip(lr_dirs, hr_dirs):
        hr = cv2.imread(hr_dir)
        lr = Image.open(lr_dir).convert('RGB')
        lr = trans(lr).unsqueeze(0).to('cuda:0')
        _, _, h_old, w_old = lr.size()

        pred = model(lr, [int(scale * h_old), int(scale * w_old)])

        pred = pred[0]
        pred_dir = os.path.join(args.save_path, os.path.splitext(os.path.basename(lr_dir))[0] + f'{args.model_name}_x{args.scale_max}' +
                                os.path.splitext(os.path.basename(lr_dir))[1])
        transform.ToPILImage()(pred.cpu()).save(pred_dir)

        pred = cv2.imread(pred_dir)
        psnr = calculate_psnr(hr, pred, border=int(args.scale_max))
        performance['PSNR'][os.path.basename(lr_dir)] = psnr
        ssim = calculate_ssim(hr, pred, border=int(args.scale_max))
        performance['SSIM'][os.path.basename(lr_dir)] = ssim
        print(f'{os.path.basename(lr_dir)} psnr={psnr} ssim={ssim}')

    performance['PSNR_AVG'] = sum(performance['PSNR'].values())/len(performance['PSNR'])
    performance['SSIM_AVG'] = sum(performance['SSIM'].values()) / len(performance['SSIM'])
    print('PSNR=',performance['PSNR_AVG'],'SSIM=',performance['SSIM_AVG'])
    with open(f'{args.save_path}/ucmx{args.scale_max}.json', 'w') as j:
        json.dump(performance, j, indent=True, sort_keys=False, skipkeys=True)
