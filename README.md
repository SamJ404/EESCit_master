# EESCit
Edge Enhanced and Spatial-Channel Information Enhanced Arbitrary-scale Super-resolution for Remote Sensing Image
This is the offical pytorch implementation of EESCit. [[Pretrained models]](https://pan.baidu.com/s/1QXDVf1FX790EjhlTGBUARg) (extraction code:gdf3) are available. Feel free to send emails to SAMantha404@163.com, discussion is welcome 🙌:.  
## Contents 📖 ##  
***
>[Brief Introduction](#section1)  
>[Quick Start](#section2)  
>
>>[Training](#section21)  
>>[Testing](#section22)  
>>
>[Detailed Structure of EESCit](#section3)  
>[Results](#section4)  
>>[Quantitative Results](#section41)  
>>[Visual Results](#section42)
>>
>[Copyright](#section5)
<a id='section1'></a>
## Brief Introduction ✨ ##
***
* **Motivation:** Arbitrary-scale super-resolution (ASSR) for remote sensing (RS) images suffered from the challenge of the low signal-to-noise ratio (SNR) and sever degradation of high-frequency details. We proposed EESCit with the capability of enhancing edge feature and sensitivity of valid signal information.
* **Innovation:** An Edge Enhanced Module (EEM) with echo-shaped feature propagation pathways and adaptively parameterized Sobel operators is developed to achieve edge-aware feature recalibration, preserving structural continuity in edge regions. In addition, a cascaded framework combining spatial and channel information enhancement operations (SCEM) is incorporated into the algorithm to suppress interference noise and enhance discriminative features in RS imagery.
* **Keywords:** Arbitrary-scale super-resolution, Remote sensing imagery, Implicit transformer, Edge enhancement, Learnable Soble operators
<a id='section2'></a>
## Quick Start ✨ ##  
***
<a id='section21'></a>
### Training 💪 ###  
* Prepare environment:  
```
pip install -r requirement.txt
```  
* Prepare dataset information text file:
To generate HR images information for train dataset (Remember to 'hash' the irrelevant parts according to the annotation):  
```
python generate_meta_info.py --dataset_dir [PATH/TO/DATASET1,PATH/TO/DATASET2] --save_dir PATH/TO/SAVE --txt_name META_INFO.txt
```
To generate HR-LR image pair information for validate dataset (Remember to 'hash' the irrelevant parts according to the annotation):  
```
python generate_meta_info.py --hrset_dir [PATH/TO/HRIMGS1,PATH/TO/HRIMGS2] --lrdet_dir [PATH/TO/LRIMGS1,PATH/TO/LRIMGS2] --save_dir PATH/TO/SAVE --txt_name META_INFO.txt
```
* Modify `./configs/train_x_EESCit_bestval.yaml` before start training, where x means backbone. Take `train_swin_EECSit_bestval.yaml` for example, pay extra attention to parameters:
```
train_dataset['dataset']['args']['split_file']: directory to hr images meta info txt file
train_dataset['wrapper']['args']['scales']: training scales (the scales used will become in-scale)
train_dataset['batch_size']: training batch size
val_dataset['dataset']['args']['meta_file']: directory to hr-lr image pairs meta info txt file
val_dataset['wrapper']['args']['scale']: validate scale, consistent with hr-lr image pairs
epoch_max: total training epochs
epoch_val: validate every x epochs
epoch_save: save every x epochs
```
* Train EESCit model:  
```
python train_EESCit_bestval.py --config ./configs/train_x_EESCit_bestval.yaml
```  
<a id='section22'></a>
### Testing 💪 ###  
We provide EESCit checkpoints with `swin-transformer` and `rdn` backbones pretrained on [[UCMerced_LandUse]](https://pan.baidu.com/s/1QXDVf1FX790EjhlTGBUARg), [[AID]](https://pan.baidu.com/s/1QXDVf1FX790EjhlTGBUARg) and [[NWPU_RESISC45]](https://pan.baidu.com/s/1QXDVf1FX790EjhlTGBUARg). (extraction code: gdf3)  
* Download the checkpoints to `./checkpoints`. (Other folders you want, 'checkpoints' is recommended.)
* Inference on dataset UCMerced_LandUse
```
python test_UCM.py --model /PATH/TO/CHECKPOINT.pth --model_name NAME --scale_max SCALE --meta_info PATH/TO/META_INFO.TXT --save_path /PATH/TO/SAVE/IMGS/AND/METRICS.JSON
```
The `--scale_max` and scale of hr-lr in meta_info.txt need to be matched.
<a id='section3'></a>
## Detialed Structure of EESCit ✨ ##
***
* **Overview:** The proposed EESCit model is based on implicit transformer structure, utilizing implicit coordinate mapping function to realize continuous magnification. The overview of our algorithm is illustrated in Fig 1. The network follows a two-stage inference framework including feature extraction and arbitrary-scale reconstruction based on implicit transformer. The query branch implements dual-stage edge enhance module (EEM), where the primary enhancement operates on native-resolution feature maps from the backbone encoder, while the secondary refinement processes upsampled representations generated through implicit neural coordinate transformation. The key-value dual pathways share the same spatial-channel cascading enhance module (SCEM) operating on backbone-extracted feature, followed by branch-diverged processing where implicit coordinate transformations precede secondary SCEM operations in respective pathways.
![Overview Sturcture](https://github.com/SamJ404/EESCit_master/blob/main/illustration/EESCit.jpg)
*Figure 1*
* **EEM:** We proposed an Edge Enhance Module (EEM) implementing Echo Forward Extraction (EFE) operation with novel structure and Quad-directional Edge Focus (QEF) based on learnable Sobel convolution for better feature propagation and edge saliency preservation in low-SNR conditions. The construction of our EEM is depicted as Fig 2.
![Edeg Enhance Module](https://github.com/SamJ404/EESCit_master/blob/main/illustration/EEM.jpg)
*Figure 2*
<a id='section4'></a>
## Results ✨ ##  
We conducted comparative experiments on UCMerced_LandUse dataset comparing our approach with other SOTA ASSR algorithms. We retrained all the models applying exactly the same training configuration for a fair evaluation.  
<a id='section41'></a>
### Quantitative Results 👀 ###  
***
The in-scale quantitative results including PSNR and SSIM metrics are illustrated in table below:
![quantitative_table in-scale](https://github.com/SamJ404/EESCit_master/blob/main/illustration/in_scale.png)  
The out-scale with integer value quantitative results including PSNR and SSIM metrics are illustrated in table below:
![quantitative table out-scale-int](https://github.com/SamJ404/EESCit_master/blob/main/illustration/out_scale_int.png)
The out-scale with continuous value quantitative results including PSNR and SSIM metrics are illustrated in table below:
![quantitative table ou-scale-continuous](https://github.com/SamJ404/EESCit_master/blob/main/illustration/out_scale_continuous.png)
<a id='section42'></a>
### Visual Results 👀 ###  
***
Take image 'storagetanks01' for example, we depicted the x8, x10 and x12 reconstruction mission as follows:
![storagetanks01x8](https://github.com/SamJ404/EESCit_master/blob/main/illustration/storagetanks01x8.jpg)
![storagetanks01x10](https://github.com/SamJ404/EESCit_master/blob/main/illustration/storagetanks01x10.jpg)
![storagetanks01x12](https://github.com/SamJ404/EESCit_master/blob/main/illustration/storagetanks01x12.jpg)
<a id='section5'></a>
## Copyright ✨ ##
The copyright of this repository and attachments belongs to SamanthaJiao, if you have any question please contact me! The paper is in process.
