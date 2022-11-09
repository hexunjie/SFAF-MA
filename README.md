# SFAF-MA:Spatial Feature Aggregation Fusion with Modality Adaptation for RGB-Thermal Semantic Segmentation
This is the overall architecture of our work, which contains two encoder branches and one decoder branch. Besides, three subsections which are Modality Difference Adaptive Fusion (MDAF), Spatial Semantic Fusion (SSF) and the Decoder Block are shown.

![](https://github.com/hexunjie/SFAF-MA/blob/main/pictures/02.jpg)

## Preparation
Build the environment with `python3.6.5` and `torch1.8.0`.

The dataset is the public [MFNet]() dataset and [PST900]() dataset.

## Implement Demo
The `.pth` file can be downlodad from [final.pth]()

The path of dataset and the weight() should be changed to your path.

    cd SFAF-MA
    python run_demo.py

After which the result can be re-implemented as follows.

![](https://github.com/hexunjie/SFAF-MA/blob/main/pictures/03.png)
![](https://github.com/hexunjie/SFAF-MA/blob/main/pictures/04.png)
Our method clearly achieves the best performance with 69.5% mAcc and 55.3% mIoU values in MFNet dataset and 65.7% mAcc and 60.0% mIoU values in PST900 dataset, which are both the best results in each dataset. 

## Train
In the training process, you may need to adjust the parameters to adapt to your device, such as the batch size and the learning rate.

To train it,

    cd SFAF-MA
    python train.py
    
