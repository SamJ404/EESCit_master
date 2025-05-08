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
