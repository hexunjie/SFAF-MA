# SFAF-MA:Spatial Feature Aggregation Fusion with Modality Adaptation for RGB-Thermal Semantic Segmentation
This is the overall architecture of our work, which contains two encoder branches and one decoder branch. Besides, three subsections which are Modality Difference Adaptive Fusion (MDAF), Spatial Semantic Fusion (SSF) and the Decoder Block are shown.

![](https://github.com/hexunjie/SFAF-MA/blob/main/pictures/02.png)

## Preparation
Build the environment with `python3.6.5` and `torch1.8.0`.

The dataset is the public MFNet dataset and it can be got from [here](https://store4.gofile.io/download/direct/744d63d7-d4af-44ff-9b45-27d54f3eb697/dataset.zip).

## Implement Demo
The `.pth` file can be downlodad from [final.pth](https://store4.gofile.io/download/direct/161d7392-0e9f-4093-8afd-eb98380394cb/199.pth)

The path of dataset and the weight(final.pth) should be changed to your path.

    cd SFAF-MA
    python run_demo.py

After which the result can be re-implemented as follows.

![](https://github.com/hexunjie/SFAF-MA/blob/main/pictures/03.png)
Our method clearly achieves the best performance with 69.1% mAcc and 55.9% mIoU values, which are 5.0% and 2.7% higher than those of RTFNet, respectively. 

## Train
In the training process, you may need to adjust the parameters to adapt to your device, such as the batch size and the learning rate.

To train it,

    cd SFAF-MA
    python train.py
    
