import torch
import numpy as np
import matplotlib.pyplot as plt
lenet_l1=torch.load('./best_iter_firstbatch_mnist_lenet_L1.pt')
standard_l1=torch.load('./best_iter_firstbatch_mnist_standard_L1.pt')
ddn_l1=torch.load('./best_iter_firstbatch_mnist_ddn_L1.pt')

lenet_l2_MPBA=torch.load('./best_iter_firstbatch_mnist_lenet_L2_2000genetation_MBPA_popsize16.pt')
standard_l2_MPBA=torch.load('./best_iter_firstbatch_mnist_standard_L2_2000genetation_MBPA_popsize16.pt')
ddn_l2_MPBA=torch.load('./best_iter_firstbatch_mnist_ddn_L2_2000genetation_MBPA_popsize16.pt')

lenet_l2_MPBA_AdaStep=torch.load('./best_iter_firstbatch_mnist_lenet_L2_2000genetation_MBPAAdaStep_popsize16.pt')
standard_l2_MPBA_AdaStep=torch.load('./best_iter_firstbatch_mnist_standard_L2_2000genetation_MBPAAdaStep_popsize16.pt')
ddn_l2_MPBA_AdaStep=torch.load('./best_iter_firstbatch_mnist_ddn_L2_2000genetation_MBPAAdaStep_popsize16.pt')

x = np.linspace(0, 2000, 2000)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x, lenet_l1,color='r',label='LeNet5-M_MPBA-L1')
ax1.plot(x, standard_l1,color='b',label='Standard-M_MPBA-L1')
ax1.plot(x, ddn_l1,color='g',label='DDN-M_MPBA-L1')

ax2.plot(x, lenet_l2_MPBA,color='violet',label='LeNet5-M_MPBA-L2',ls='--')
ax2.plot(x, standard_l2_MPBA,color='royalblue',label='Standard-M_MPBA-L2',ls='--')
ax2.plot(x, ddn_l2_MPBA,color='tan',label='DDN-M_MPBA-L2',ls='--')

ax2.plot(x, lenet_l2_MPBA_AdaStep,color='navy',label='LeNet5-M_MPBA_AdaStep-L2',ls='-.')
ax2.plot(x, standard_l2_MPBA_AdaStep,color='springgreen',label='Standard-M_MPBA_AdaStep-L2',ls='-.')
ax2.plot(x, ddn_l2_MPBA_AdaStep,color='purple',label='DDN-M_MPBA_AdaStep-L2',ls='-.')

plt.legend(loc="best")
plt.title("Fitness curve of attack on MNIST")

ax1.set_xlabel('Generation Num', fontdict={'size': 10})
ax1.set_ylabel('average minus fitness of elites(L1 attack)', fontdict={'size': 10})
ax1.legend(bbox_to_anchor=(0.1,1),fontsize='small')
ax2.set_ylabel('average minus fitness of elites(L2 attack)', fontdict={'size': 10})
ax2.legend(bbox_to_anchor=(0.9,1),fontsize='small')

#plt.grid(True)
print('finish')




