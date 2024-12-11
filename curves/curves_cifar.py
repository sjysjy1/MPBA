import torch
import numpy as np
import matplotlib.pyplot as plt
cifar_wong_L1=torch.load('./best_iter_firstbatch_cifar_wong_L1_2000generation.pt')
cifar_standard_L1_1000=torch.load('./best_iter_firstbatch_cifar_standard_L1_1000generation.pt')
cifar_standard_L1_2000=torch.load('./best_iter_firstbatch_cifar_standard_L1_2000generation.pt')
cifar_standard_L1_3000=torch.load('./best_iter_firstbatch_cifar_standard_L1_3000generation.pt')

cifar_standard_MPBA_L2=torch.load('./best_iter_firstbatch_cifar_standard_L2_MPBA_4000generation_popsize32.pt')
cifar_wong_MPBA_L2=torch.load('./best_iter_firstbatch_cifar_WongLinf_L2_MPBA_3000generation_popsize32.pt')#ok

cifar_standard_MPBAAdapStep_L2=torch.load('./best_iter_firstbatch_cifar_standard_L2_MPBAAdaStep_3000generation.pt') #ok
cifar_wong_MPBAAdapStep_L2=torch.load('./best_iter_firstbatch_cifar_WongLinf_L2_MPBAAdapStep_4000generation_popsize32.pt')#ok

x_1000 = np.linspace(0, 1000, 1000)
x_2000 = np.linspace(0, 2000, 2000)
x_3000 = np.linspace(0, 3000, 3000)
x_4000 = np.linspace(0, 4000, 4000)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x_1000, cifar_standard_L1_1000,color='r',label='Standard-C-1000-MPBA-L1')
ax1.plot(x_2000, cifar_standard_L1_2000,color='b',label='Standard-C-2000-MPBA-L1')
ax1.plot(x_3000, cifar_standard_L1_3000,color='violet',label='Standard-C-3000-MPBA-L1')
ax1.plot(x_2000, cifar_wong_L1,color='g',label='WongLinf-C-MPBA-L1')

ax2.plot(x_4000, cifar_standard_MPBA_L2,color='royalblue',label='Standard-C_MPBA-L2',ls='--')
ax2.plot(x_3000, cifar_wong_MPBA_L2,color='tan',label='WongLinf-C_MPBA-L2',ls='--')
ax2.plot(x_3000, cifar_standard_MPBAAdapStep_L2,color='springgreen',label='Standard-C_MPBA_AdaStep-L2',ls='-.')
ax2.plot(x_4000, cifar_wong_MPBAAdapStep_L2,color='purple',label='WongLinf-C_MPBA_AdaStep-L2',ls='-.')

ax1.set_xlabel('Generation Num', fontdict={'size': 10})
ax1.set_ylabel('average minus fitness of elites(L1 attack)', fontdict={'size': 10})
ax1.legend(bbox_to_anchor=(0.1,1),fontsize='small')
ax2.set_ylabel('average minus fitness of elites(L2 attack)', fontdict={'size': 10})
ax2.legend(bbox_to_anchor=(1.0,1),fontsize='small')

#plt.legend(loc="best")
plt.title("Fitness curve of attack on CIFAR10")
#plt.grid(True)
print('finish')