import os
import imageio

path='/home/hxj/PST900_RGBT_Dataset/generation'
newpath='/home/hxj/PST900_RGBT_Dataset/train/images'
dirs=os.listdir(path)

for file in dirs:
    pic_dir=os.path.join(path,file)
    for i in os.listdir(pic_dir):
        image_dir=os.path.join(pic_dir,i)
        image_dir1 = image_dir[0]
        image_dir2 = image_dir[1]
        img1=imageio.imread(image_dir1)
        img2 = imageio.imread(image_dir2)
        print(img1)
