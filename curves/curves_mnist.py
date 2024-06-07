import torch
import numpy as np
import matplotlib.pyplot as plt
lenet=torch.load('./best_iter_firstbatch_mnist_lenet_L1.pt')
standard=torch.load('./best_iter_firstbatch_mnist_standard_L1.pt')
ddn=torch.load('./best_iter_firstbatch_mnist_ddn_L1.pt')


x = np.linspace(0, 2000, 2000)
plt.figure(figsize=(8, 6))
plt.plot(x, lenet,color='r',label='LeNet5-M')
plt.plot(x, standard,color='b',label='Standard-M')
plt.plot(x, ddn,color='g',label='DDN-M')
plt.legend(loc="best")
plt.xlabel('Generation Num')
plt.ylabel('average minus fitness of elites')
plt.title("Fitness curve of L1 attack on MNIST")
plt.grid(True)
print('finish')


