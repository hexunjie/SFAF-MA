# SFAF-MA:Spatial Feature Aggregation Fusion with Modality Adaptation for RGB-Thermal Semantic Segmentation
This is the overall architecture of our work, which contains two encoder branches and one decoder branch. Besides, three subsections which are Modality Difference Adaptive Fusion (MDAF), Spatial Semantic Fusion (SSF) and the Decoder Block are shown.

![](https://github.com/hexunjie/SFAF-MA/blob/main/pictures/01.png)

## Preparation
Build the environment with `python3.6.5` and `torch1.8.0`.

The dataset is the public [MFNet](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) dataset and [PST900](https://drive.google.com/open?id=1hZeM-MvdUC_Btyok7mdF00RV-InbAadm) dataset.

## Implement Demo

The path of dataset and the weight should be changed to your path.

    cd SFAF-MA
    python run_demo.py


The best `.pth` file in MFNet can be downlodad below.
 Network | Weights
------------- | -------------
 SFAF-MA-50 | [final.pth](https://drive.google.com/drive/folders/15BnNBZTEX9nPcZNtCq0IzMjE1IFDvayg)
 SFAF-MA-101 | [final.pth](https://drive.google.com/drive/folders/1y3xqa9hFsz8d6yiARmbhFDBTfv6p644B)
 SFAF-MA-152 | [final.pth](https://drive.google.com/drive/folders/1h3hN8T0ae0bVKliCCI0upng__9Ux1B7q)
 
After which the result can be re-implemented as follows.

![](https://github.com/hexunjie/SFAF-MA/blob/main/pictures/02.png)

![](https://github.com/hexunjie/SFAF-MA/blob/main/pictures/03.png)

Our method clearly achieves the best performance with 69.6% mAcc, 55.5% mIoU, 68.0% mPre and 67.0% F1-score values in MFNet dataset, and 68.3% mAcc, 54.5% mIoU, 75.7% mPre and 65.8% F1-score values in PST900 dataset, which are both the best results in each dataset. 

## Train
In the training process, you may need to adjust the parameters to adapt to your device, such as the batch size and the learning rate.

To train it,

    cd SFAF-MA
    python train.py
    
