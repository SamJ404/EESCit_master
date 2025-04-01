# Originally by SAM J, Authorized by SAM J

import os
from glob import glob
import argparse
import cv2

parser = argparse.ArgumentParser()
# used in gerenating lr images
parser.add_argument('--dataset_dir', type=list, default=['/PATH/TO/HRIMGS','PATH/TO/middle_resized_imgs', '/PATH/TO/LRIMGS'])

parser.add_argument('--hrset_dir', type=list, default=['/PATH/TO/HRIMGS'])
parser.add_argument('--lrset_dir', type=list, default=['/PATH/TO/LRIMGS'])
parser.add_argument('--save_dir', type=str, default='/PATH/TO/SAVE')
parser.add_argument('--txt_name', type=str, default='meta_info.txt')
args = parser.parse_args()

# generate single img meta info
# img_paths = []
# for dataset in args.dataset_dir:
#     for cls in sorted(os.listdir(dataset)):
#         cls_dir = os.path.join(dataset, cls)
#         img_paths.append(sorted(glob(os.path.join(cls_dir, '*.png'))))
# img_paths = sum(img_paths, [])
# txt_path = os.path.join(args.save_dir, args.txt_name)
# with open(txt_path, 'w') as txt_file:
#     # for img_path in img_paths:
#     #     img = cv2.imread(img_path)
#     #     if img.shape == (256,256,3):
#     #         txt_file.write(f'{img_path}\n')
#     for img_path in img_paths:
#         txt_file.write(f'{img_path}\n')
#     txt_file.close()

# # generate pair img meta info
txt_path = os.path.join(args.save_dir, args.txt_name)
with open(txt_path, 'w') as txt_file:
    for hrset, lrset in zip(args.hrset_dir, args.lrset_dir):
        for cls in sorted(os.listdir(hrset)):
            hr_imgs = sorted(glob(os.path.join(hrset, cls, '*.png')))
            lr_imgs = sorted(glob(os.path.join(lrset, cls, '*.png')))
            for hr_img, lr_img in zip(hr_imgs, lr_imgs):
                hr = cv2.imread(hr_img)
                lr = cv2.imread(lr_img)
                if hr.shape == (253,253,3) and lr.shape == (22,22,3):
                    # print(hr_img, lr_img)
                    txt_file.write(f'{lr_img},{hr_img}\n')
                # txt_file.write(f'{lr_img}\n')
    txt_file.close()


# generate lr img
# datasethr_dir = args.dataset_dir[0]
# # dataset253hr_dir = args.dataset_dir[1]
# datasetlr_dir = args.dataset_dir[-1]
# for cls in sorted(os.listdir(datasethr_dir)):
#     cls_dir = os.path.join(datasethr_dir, cls)
#     os.makedirs(os.path.join(datasetlr_dir, cls), exist_ok=True)
#     # os.makedirs(os.path.join(dataset253hr_dir, cls), exist_ok=True)
#     for hr_dir in sorted(glob(os.path.join(cls_dir, '*.png'))):
#         hr_name = hr_dir.split('/')[-1]
#         lr_dir = os.path.join(datasetlr_dir, cls, hr_name)
#         # hr253_dir = os.path.join(dataset253hr_dir, cls, hr_name)
#         hr = cv2.imread(hr_dir)
#         # if hr.shape == (256, 256, 3):
#         #     hr253 = hr[1:254, 1:254, :]
#         #     cv2.imwrite(hr253_dir, hr253)
#         lr = cv2.resize(hr, dsize=[22, 22], interpolation=cv2.INTER_LINEAR)
#         cv2.imwrite(lr_dir, lr)
#     print(f'finish processing {cls}')
