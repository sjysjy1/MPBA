import torch
import numpy as np
import matplotlib.pyplot as plt

imagenet_WongLinf_L1=torch.load('best_iter_firstbatch_ImageNet_WongLinf_L1_MPBA_2000generation.pt')
imagenet_WongLinf_L2_MPBAAdapStep=torch.load('best_iter_firstbatch_ImageNet_WongLinf_L2_MBPAAdaStep_8000generation_popsize512.pt')
imagenet_WongLinf_L2_MPBAAdapStep_sigma=torch.load('best_iter_firstbatch_ImageNet_WongLinf_L2_MBPAAdaStep_8000generation_popsize512_sigma.pt')
x_2000 = np.linspace(0, 2000, 2000)
x_8000=np.linspace(0, 8000, 8000)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x_2000, imagenet_WongLinf_L1,color='g',label='WongLinf-I-MPBA-L1')
ax2.plot(x_8000, imagenet_WongLinf_L2_MPBAAdapStep,color='royalblue',label='WongLinf-I_MPBA_AdaStep-L2',ls='--')
#ax2.plot(x_8000, imagenet_WongLinf_L2_MPBAAdapStep_sigma,color='b',label='WongLinf-I_MPBA_AdaStep-L2')

ax1.set_xlabel('Generation Num', fontdict={'size': 10})
ax1.set_ylabel('average minus fitness of elites(L1 attack)', fontdict={'size': 10})
ax1.legend(bbox_to_anchor=(0.1,1),fontsize='small')
ax2.set_ylabel('average minus fitness of elites(L2 attack)', fontdict={'size': 10})
ax2.legend(bbox_to_anchor=(1.0,1),fontsize='small')

#plt.legend(loc="best")
plt.title("Fitness curve of attack on ImageNet")
#plt.grid(True)
print('finish')