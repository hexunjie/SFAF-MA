import pandas
import seaborn as sns
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

"""
names = ['unlabelled','car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump']
os.chdir(os.getcwd())
data = pandas.read_table('1.txt', sep='\s+')
data=[[99.0418, 93.8277,  82.1874, 74.6426, 67.1410, 27.7180,  23.2213, 54.0717,  75.5767],
[98.9928, 94.0947,  82.5558, 74.8534, 67.3594, 25.2720,  17.5115, 54.3169,  75.5297],
[99.0751, 93.9140,  81.5911, 73.4543, 66.6091, 24.9427,  19.6471, 54.2055,  74.1682],
[99.1325, 93.6512,  81.1874, 75.6426, 67.2410, 27.7280,  23.2113, 54.1717,  75.6767],
[98.4565, 94.4947,  82.4558, 74.8534, 67.6494, 25.2730,  17.5115, 54.3169,  75.5297],
[99.0751, 93.9140,  81.5911, 73.4543, 66.6091, 24.9427,  19.6471, 54.2055,  74.1682],
[99.0418, 93.8277,  82.1874, 74.6426, 67.1410, 27.7180,  23.2213, 54.0717,  75.5767],
[98.9928, 94.0947,  82.5558, 74.8534, 67.3594, 25.2720,  17.5115, 54.3169,  75.5297],
[99.0751, 93.9140,  81.5911, 73.4543, 66.6091, 24.9427,  19.6471, 54.2055,  74.1682],
[99.1325, 93.6512, 81.1874, 75.6426, 67.2410, 27.7280, 23.2113, 54.1717, 75.6767],
[98.4565, 94.4947, 82.4558, 74.8534, 67.6494, 25.2730, 17.5115, 54.3169, 75.5297]]

#print(data.shape)
data_dict={}
for col, gf_lst in zip(names,data):
    data_dict[col]=gf_lst
data_df=pandas.DataFrame(data_dict)
cor1=data_df.corr()
#data=pandas.DataFrame({'a':torch.randn(1,100000),'b':torch.randn(1,100000)})
#cor=data.corr()
print(cor1)
correction=abs(cor1)
fig=plt.figure()
ax=plt.subplots(figsize=(9,9))
ax=sns.heatmap(correction, annot=False, square=True, cmap='Blues')
plt.xticks(np.arange(9)+0.5,names)
plt.yticks(np.arange(9)+0.5,names)
ax.set_title('prediction correlation')
plt.savefig('prediction.tif',dpi=300)
plt.show()
"""

rgb= torch.randn(64,400)
t=torch.randn(64,400)
output=np.corrcoef(rgb,t)
print(output.shape)
fig=plt.figure()
ax=plt.subplots(figsize=(9,9))
ax=sns.heatmap(output, annot=False, square=True, cmap='Blues')
plt.xticks(np.arange(0)+0.5)
plt.yticks(np.arange(0)+0.5)
ax.set_title('test')
plt.savefig('test.tif',dpi=300)
plt.show()