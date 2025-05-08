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
>[Detailed Structure of SymSwin](#section3)  
>[Results](#section4)  
>>[Quantitative Results](#section41)  
>>[Visual Results](#section42)
>>
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
* Modify `./configs/train_x_EESCit_bestval.yaml` before start training. Take `train_swin_EECSit_bestval.yaml` for example, pay extra attention to parameters:
```
train_dataset['dataset']['args']['split_file']: directory to hr images meta info txt file
train_dataset['wrapper']['args']['scales']: training scales (the scales used will become in-scale)
train_dataset['batch_size']: training batch size
val_dataset['dataset']['args']['meta_file']: directory to hr-lr image pairs meta info txt file
val_dataset['wrapper']['args']['scale']: validate scale, consistent with hr-lr image pairs
```





