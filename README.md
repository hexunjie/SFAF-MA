# SFAF-MA:Spatial Feature Aggregation Fusion with Modality Adaptation for RGB-Thermal Semantic Segmentation
This is the overall architecture of our work, which contains two encoder branches and one decoder branch. Besides, three subsections which are Modality Difference Adaptive Fusion (MDAF), Lateral Semantic Fusion (LSF) and the Decoder Block are shown.

The video which introduces the paper can be downloaded [here](https://drive.google.com/file/d/14D05Jt6IRwvf5JpcFKdSzWLjpyxYzbpn/view?usp=sharing).

![](https://github.com/hexunjie/SFAF-MA/blob/main/pictures/02.jpg)

## Preparation
Build the environment with `python3.6.5` and `torch1.9.0`.

The dataset is the public MFNet dataset and it can be got from [here](https://drive.google.com/file/d/17H6Oj_q-EqAT1ebj3bi7QS4OhQWxuEJh/view?usp=sharing).

## Implement Demo
The `.pth` file can be downlodad from [final.pth](https://drive.google.com/file/d/1oVKuanZTmJ896Yx3wiCq1-e4PjbP2e2S/view?usp=sharing)

The path of dataset and the weight(final.pth) should be changed to your path.

    cd SFAF-MA
    python run_demo.py

After which the result can be re-implemented as follows.

![](https://github.com/hexunjie/SFAF-MA/blob/main/pictures/3.png)

## Train
In the training process, you may need to adjust the parameters to adapt to your device, such as the batch size and the learning rate.

To train it,

    cd SFAF-MA
    python train.py
    
