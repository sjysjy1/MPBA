import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model_MNIST import model_MNIST
from LeNet5 import LeNet5
import time
from datetime import datetime
import random
import matplotlib.pyplot as plt
import torchattacks

from functools import partial
from adv_lib.attacks import sigma_zero,alma, ddn,fmn,fab
from adv_lib.utils.attack_utils import run_attack
from adv_lib.utils.lagrangian_penalties import all_penalties
from ZOO_Attack_PyTorch import zoo_l2_attack_black

import sys
#sys.path.append('./sparse-rs')
sys.path.append('./blackbox_adv_examples_signhunter/src')

from my_gene import MPBA,MPBA_adapt

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()

seed_torch()
device=('cuda:0' if torch.cuda.is_available() else 'cpu')
#device='cpu'
print(device)

test_dataset=datasets.MNIST(root='./data',train=False,download=True,transform=transforms.ToTensor())

model=model_MNIST()
model.to(device)
criterion=nn.CrossEntropyLoss()
batch_size=512
#batch_size=8
list_para=[

{'model':'LeNet5','attack':'MPBA','p_norm':'L1','generation_num':2000,'popsize':16,'init_pop_range':1.0,'stepsize_start':0.2,'stepsize_end':0.01,'a':None,'rate_mutation':0.1,'rate_crossover':0,'batch_size':batch_size,'mu_start':0.1,'mu_end':100.0,'lower':0.1,'upper':1.0,'c':1},
{'model':'Standard','attack':'MPBA','p_norm':'L1','generation_num':2000,'popsize':16,'init_pop_range':1.0,'stepsize_start':0.2,'stepsize_end':0.01,'a':None,'rate_mutation':0.1,'rate_crossover':0,'batch_size':batch_size,'mu_start':1,'mu_end':100.0,'lower':1,'upper':1,'c':1},
{'model':'ddn','attack':'MPBA','p_norm':'L1','generation_num':2000,'popsize':16,'init_pop_range':1.0,'stepsize_start':0.2,'stepsize_end':0.01,'a':None,'rate_mutation':0.1,'rate_crossover':0,'batch_size':batch_size,'mu_start':10,'mu_end':100.0,'lower':0.1,'upper':1,'c':1},

{'model': 'LeNet5','attack':'EAD','batch_size':batch_size ,'iter_num':1000},
{'model': 'Standard','attack':'EAD','batch_size':batch_size ,'iter_num':1000},
{'model': 'ddn','attack':'EAD','batch_size':batch_size ,'iter_num':1000},

{'model':'LeNet5','attack':'FMN','p_norm':1,'steps':1000,'batch_size':batch_size,'γ_init':0.05,'α_init':0.01},
{'model':'Standard','attack':'FMN','p_norm':1,'steps':1000,'batch_size':batch_size,'γ_init':0.05,'α_init':0.1},
{'model':'ddn','attack':'FMN','p_norm':1,'steps':1000,'batch_size':batch_size,'γ_init':0.05,'α_init':0.1},

{'model': 'LeNet5','attack':'ALMA','p_norm':'l1','batch_size':batch_size ,'iter_num':1000,'init_lr_dist':0.5},
{'model': 'Standard','attack':'ALMA','p_norm':'l1','batch_size':batch_size ,'iter_num':1000,'init_lr_dist':0.5},
{'model': 'ddn','attack':'ALMA','p_norm':'l1','batch_size':batch_size ,'iter_num':1000,'init_lr_dist':0.5},



{'model':'LeNet5','attack':'MPBA','p_norm':'L2','generation_num':2000,'popsize':16,'init_pop_range':1.0,'stepsize_start':0.1,'stepsize_end':0.01,'a':3,'rate_mutation':0.1,'rate_crossover':0,'batch_size':batch_size,'mu_start':0.01,'mu_end':0.1,'lower':1.0,'upper':10.0,'c':1},#MPBA
{'model':'Standard','attack':'MPBA','p_norm':'L2','generation_num':2000,'popsize':16,'init_pop_range':1.0,'stepsize_start':0.2,'stepsize_end':0.01,'a':3,'rate_mutation':0.1,'rate_crossover':0,'batch_size':batch_size,'mu_start':0.1,'mu_end':10.0,'lower':0.1,'upper':1,'c':1},
{'model':'ddn','attack':'MPBA','p_norm':'L2','generation_num':2000,'popsize':16,'init_pop_range':1.0,'stepsize_start':0.2,'stepsize_end':0.01,'a':3,'rate_mutation':0.1,'rate_crossover':0,'batch_size':batch_size,'mu_start':1,'mu_end':100.0,'lower':0.1,'upper':1,'c':1},

{'model':'LeNet5','attack':'MPBA_ada_step','p_norm':'L2','generation_num':2000,'popsize':16,'init_pop_range':1.0,'stepsize_start':0.2,'rate_mutation':0.1,'rate_crossover':0,'batch_size':batch_size,'mu_start':0.1,'mu_end':100.0,'lower':0.1,'upper':1.0,'c':1},
{'model':'Standard','attack':'MPBA_ada_step','p_norm':'L2','generation_num':2000,'popsize':16,'init_pop_range':1.0,'stepsize_start':0.2,'rate_mutation':0.1,'rate_crossover':0,'batch_size':batch_size,'mu_start':0.1,'mu_end':10.0,'lower':0.1,'upper':1,'c':1},
{'model':'ddn','attack':'MPBA_ada_step','p_norm':'L2','generation_num':2000,'popsize':16,'init_pop_range':1.0,'stepsize_start':0.2,'rate_mutation':0.1,'rate_crossover':0,'batch_size':batch_size,'mu_start':1,'mu_end':100.0,'lower':0.1,'upper':1,'c':1},


{'model':'LeNet5','attack':'ZOO','solver':'adam','targeted':False,'use_tanh':True},
{'model':'Standard','attack':'ZOO','solver':'adam','targeted':False,'use_tanh':True},
{'model':'ddn','attack':'ZOO','solver':'adam','targeted':False,'use_tanh':True},

{'model':'LeNet5','attack':'SquareAttack','p_norm':'L2','eps':1,'steps':40000,'batch_size':batch_size},
{'model':'Standard','attack':'SquareAttack','p_norm':'L2','eps':2,'steps':40000,'batch_size':batch_size},
{'model':'ddn','attack':'SquareAttack','p_norm':'L2','eps':3,'steps':40000,'batch_size':batch_size},
    ]

for item in list_para:
    list_success_fail=[]
    list_pert=[]
    print(item)
    verbose=False
    if verbose:
       empty = []
       torch.save(empty, './curvedata_MNIST_{}_{}.pt'.format(item['model'],item['attack']))
    if item['model']=='LeNet5':
        model = LeNet5()
        model.load_state_dict(torch.load('./models/mnist/MNIST_LeNet_onlyweight.pth'), False)
    else:
        model = model_MNIST()
        if item['model'] == 'Standard':
            model.load_state_dict(torch.load('./models/mnist/mnist_regular.pth'), False)
        elif item['model'] == 'ddn':
            model.load_state_dict(torch.load('./models/mnist/mnist_robust_ddn.pth'), False)  # ALM paper github : l2 adversarially trained
        elif item['model'] == 'trades':
            model.load_state_dict(torch.load('./models/mnist/mnist_robust_trades.pt'),False)  # ALM paper github: l_\infty adversarially trained
    #model.load_state_dict(torch.load('../models/mnist/model_mnist_smallcnn.pt'), False)  # model from github of trades:https://github.com/yaodongyu/TRADES
    model.to(device)
    model.eval()  # turn off the dropout
    test_data = torch.unsqueeze(test_dataset.data, dim=1)
    test_labels = test_dataset.test_labels.to(device)
    test_data_normalized = test_data / 255.0
    test_data_normalized = test_data_normalized.to(device)
    outputs = model(test_data_normalized)
    _, labels_predict = torch.max(outputs, 1)
    correct = torch.eq(labels_predict, test_labels)
    correct_sum = correct.sum()
    correct_index=[]
    for i in range(10000):
        if correct[i]:
            correct_index.append(i)
    #print(correct.sum())
    print('clean accuracy is:', correct_sum / 10000.0)
    start_time = time.time()
    if item['attack'] == 'MPBA':
        with torch.no_grad():
             for i in range(0,len(correct_index),item['batch_size']):
                print('***************{}th batch***********'.format(int(i/item['batch_size'])))

                images=test_data_normalized[correct_index[i:i+item['batch_size']]].clone().detach()
                labels=test_labels[correct_index[i:i+item['batch_size']]]


                adv_images=MPBA(model,images,labels,population_size=item['popsize'],init_pop_range=item['init_pop_range'],generation_num=item['generation_num'],p_norm=item['p_norm'],stepsize_start=item['stepsize_start'],stepsize_end=item['stepsize_end'],a=item['a'],mu_start=item['mu_start'],mu_end=item['mu_end'],lower=item['lower'],upper=item['upper'],rate_mutation_ini=item['rate_mutation'],rate_crossover_ini=item['rate_crossover'],c=item['c'],msg='curvedata_MNIST_'+item['model']+'_'+item['attack'])

                outs = model(adv_images)
                _, labels_predict = torch.max(outs, 1)
                success = (labels_predict != labels)
                list_success_fail = list_success_fail + success.tolist()

                if item['p_norm']=='L0':
                   perturbation= torch.norm((adv_images - images).view(len(images), -1), p=0, dim=1)
                elif item['p_norm'] == 'L1':
                   perturbation= torch.norm((adv_images - images).view(len(images), -1), p=1, dim=1)
                elif item['p_norm'] == 'L2':
                   perturbation= torch.norm((adv_images - images).view(len(images), -1), p=2, dim=1)
                elif item['p_norm'] == 'Linf':
                   perturbation= torch.norm((adv_images - images).view(len(images), -1), p=float('inf'), dim=1)
                list_pert = list_pert + perturbation.tolist()

                print('perturbation is: ', torch.t(perturbation))
                print('average of perturbation is:', perturbation.sum() / sum(success))
                print('success rate is:', sum(list_success_fail) / len(list_success_fail))
                end_time = time.time()
                print('time used is:',end_time-start_time)

    elif item['attack'] == 'MPBA_ada_step':
        with torch.no_grad():
             for i in range(0,len(correct_index),item['batch_size']):
                print('***************{}th batch***********'.format(int(i/item['batch_size'])))

                images=test_data_normalized[correct_index[i:i+item['batch_size']]].clone().detach()
                labels=test_labels[correct_index[i:i+item['batch_size']]]
                adv_images=MPBA_adapt(model,images,labels,population_size=item['popsize'],init_pop_range=item['init_pop_range'],generation_num=item['generation_num'],p_norm=item['p_norm'],stepsize_start=item['stepsize_start'],mu_start=item['mu_start'],mu_end=item['mu_end'],lower=item['lower'],upper=item['upper'],rate_mutation_ini=item['rate_mutation'],rate_crossover_ini=item['rate_crossover'],c=item['c'],msg='curvedata_MNIST_'+item['model']+'_'+item['attack'])
                outs = model(adv_images)
                _, labels_predict = torch.max(outs, 1)
                success = (labels_predict != labels)
                list_success_fail = list_success_fail + success.tolist()
                if item['p_norm']=='L0':
                   perturbation= torch.norm((adv_images - images).view(len(images), -1), p=0, dim=1)
                elif item['p_norm'] == 'L1':
                   perturbation= torch.norm((adv_images - images).view(len(images), -1), p=1, dim=1)
                elif item['p_norm'] == 'L2':
                   perturbation= torch.norm((adv_images - images).view(len(images), -1), p=2, dim=1)
                elif item['p_norm'] == 'Linf':
                   perturbation= torch.norm((adv_images - images).view(len(images), -1), p=float('inf'), dim=1)
                list_pert = list_pert + perturbation.tolist()

                print('perturbation is: ', torch.t(perturbation))
                print('average of perturbation is:', perturbation.sum() / sum(success))
                print('success rate is:', sum(list_success_fail) / len(list_success_fail))

    elif item['attack'] == 'EAD':
        list_const = []
        for i in range(0,len(correct_index),item['batch_size']):
            print('***************{}th batch***********'.format(int(i/item['batch_size'])))
            images=test_data_normalized[correct_index[i:i+item['batch_size']]].clone().detach()
            labels=test_labels[correct_index[i:i+item['batch_size']]]
            atk = torchattacks.EADL1(model, max_iterations=item['iter_num'])  # torchattack
            # adv_image = atk(image, label)
            adv_images = atk(images, labels)
            outs = model(adv_images)
            _, labels_predict = torch.max(outs, 1)
            success = (labels_predict != labels)
            list_success_fail = list_success_fail + success.tolist()
            perturbation = torch.norm((images - adv_images).view(len(images), -1), p=1, dim=1)
            list_pert = list_pert + perturbation.tolist()
            #print('perturbation is: ', torch.t(perturbation))
            print('average of perturbation is:', perturbation.sum() / len(perturbation))
    elif item['attack'] == 'FMN':
        method = partial(fmn, norm=item['p_norm'], steps=item['steps'], γ_init=item['γ_init'], α_init=item['α_init'])
        attack_data = run_attack(model=model, inputs=test_data_normalized[correct_index].cpu(),labels=test_labels[correct_index].cpu(), attack=method, batch_size=item['batch_size'])
        end_time = time.time()
        outs = model(attack_data['adv_inputs'].to(device))
        _, labels_predict = torch.max(outs, dim=1)
        list_success_fail = ~torch.eq(labels_predict, test_labels[correct_index])
        if item['p_norm'] == 0:
            perturbation = torch.norm((test_data_normalized[correct_index] - attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]), -1), p=0, dim=1)
        elif item['p_norm'] == 1:
            perturbation = torch.norm((test_data_normalized[correct_index] - attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]), -1), p=1, dim=1)
        elif item['p_norm'] == 2:
            perturbation = torch.norm((test_data_normalized[correct_index] - attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]), -1), p=2, dim=1)
        elif item['p_norm'] == float('inf'):
            perturbation = torch.norm((test_data_normalized[correct_index] - attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]), -1), p=float('inf'), dim=1)
        list_pert = list_pert + perturbation.tolist()
        print('perturbation is: ', perturbation)
        print('avg_pert is: ', perturbation.sum() / len(perturbation))
    elif item['attack']=='ALMA':
        penalty = all_penalties['P2']
        method=partial(alma, penalty=penalty, distance=item['p_norm'], init_lr_distance=item['init_lr_dist'], num_steps=item['iter_num'])
        attack_data = run_attack(model=model, inputs=test_data_normalized[correct_index].cpu(), labels=test_labels[correct_index].cpu(), attack=method, batch_size=item['batch_size'])
        outs = model(attack_data['adv_inputs'].to(device))
        _, labels_predict = torch.max(outs, dim=1)
        list_success_fail = ~torch.eq(labels_predict, test_labels[correct_index])
        if item['p_norm'] == 'l1':
            perturbation = torch.norm((test_data_normalized[correct_index].cpu() - attack_data['adv_inputs']).view(len(test_data_normalized[correct_index]), -1), p=1, dim=1)
        elif item['p_norm'] == 'l2':
            perturbation =torch.norm((test_data_normalized[correct_index].cpu()-attack_data['adv_inputs']).view(len(test_data_normalized[correct_index]),-1),p=2,dim=1)
        list_pert = list_pert + perturbation.tolist()
        print('perturbation is: ', perturbation)
        print('avg_pert is: ', perturbation.sum() / len(perturbation))

    elif item['attack'] == 'SquareAttack': #from package torchattacks
        with torch.no_grad():
             for i in range(0,len(correct_index),item['batch_size']):
                 print('***************{}th batch***********'.format(int(i/item['batch_size'])))
                 images=test_data_normalized[correct_index[i:i+item['batch_size']]]
                 labels=test_labels[correct_index[i:i+item['batch_size']]]
                 atk = torchattacks.Square(model, norm=item['p_norm'],eps=item['eps'],n_queries=item['steps'])  # torchattack
                 adv_images=atk(images, labels)
                 outs = model(adv_images)
                 _, labels_predict = torch.max(outs, 1)
                 success = (labels_predict != labels)
                 list_success_fail = list_success_fail + success.tolist()
                 if item['p_norm'] == 'L2':
                    perturbation = torch.norm((adv_images - images).view(len(images), -1), p=2, dim=1)
                 elif item['p_norm'] == 'Linf':
                    perturbation = torch.norm((adv_images - images).view(len(images), -1), p=float('inf'), dim=1)
                 list_pert = list_pert + perturbation.tolist()
                 print('perturbation is: ', torch.t(perturbation))
                 print('average of perturbation is:', perturbation.sum() / len(perturbation))
                 print('success rate is:', sum(list_success_fail) / len(list_success_fail))
    elif item['attack'] == 'ZOO':
        with torch.no_grad():
            list = correct_index
            test_dataset_subset = torch.utils.data.Subset(test_dataset, list)
            test_loader_subset = torch.utils.data.DataLoader(dataset=test_dataset_subset, batch_size=1,shuffle=False, num_workers=0)
            use_log = True
            for i in range(len(correct_index)//10):
                 inputs,targets=zoo_l2_attack_black.generate_data(test_loader_subset,item['targeted'],samples=1,start=i-1)
                 print("The {}th image".format(i))


                 adv_images=torch.tensor(zoo_l2_attack_black.attack(inputs-0.5, targets, model, item['targeted'], use_log, item['use_tanh'], item['solver'], device),device=device)
                 adv_images=adv_images+0.5

                 outs = model(adv_images)
                 _, labels_predict = torch.max(outs, 1)

                 if item['targeted']:
                     success = (labels_predict == torch.argmax(torch.tensor(targets)))
                 else:
                     success = (labels_predict != torch.argmax(torch.tensor(targets)))
                 list_success_fail = list_success_fail + success.tolist()

                 perturbation = torch.norm((adv_images - torch.tensor(inputs,device=device)).view(len(inputs), -1), p=2, dim=1)
                 list_pert = list_pert + perturbation.tolist()

                 print('perturbation is: ', torch.t(perturbation))
                 print('average of perturbation is:', perturbation.sum() / sum(success))
                 print('success rate is:', sum(list_success_fail) / len(list_success_fail))

    end_time = time.time()
    time_used=end_time-start_time
    print('running time:',end_time-start_time,'seconds')
    attack_success_rate=sum(list_success_fail)/correct_sum
    print('attack success rate:',attack_success_rate)
    list_pert_success=[ pert  for pert,result in zip(list_pert,list_success_fail) if result]
    avg_pert=sum(list_pert_success)/len(list_pert_success)
    print('Total avg_pert is',avg_pert)
    median_pert=torch.median(torch.tensor(list_pert_success))
    print('total median value is',median_pert)
    dict_save={'device':device,'para':item,'time_used':time_used,'list_success_fail':list_success_fail,'attack_success_rate':attack_success_rate,'list_pert':list_pert,'avg_pert':avg_pert,'median_pert':median_pert}
    if 'MPBA' == item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_pnorm_{}_GenNum{}_stepsize{}-{}_a={}_mu_{}-{}_lower{}_upper{}_c={}.pt'.format(item['model'],item['attack'],item['p_norm'],item['generation_num'],item['stepsize_start'],item['stepsize_end'],item['a'],item['mu_start'],item['mu_end'],item['lower'],item['upper'],item['c']))
    elif 'MPBA_ada_step' == item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_pnorm_{}_GenNum{}_stepsize{}_mu_{}-{}_lower{}_upper{}_c={}.pt'.format(item['model'],item['attack'],item['p_norm'],item['generation_num'],item['stepsize_start'],item['mu_start'],item['mu_end'],item['lower'],item['upper'],item['c']))
    elif 'EAD' == item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_iternum{}.pt'.format(item['model'], item['attack'], item['iter_num']))
    elif 'FMN' in item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_pnorm{}_steps{}_gammaini{}.pt'.format(item['model'], item['attack'],item['p_norm'], item['steps'],item['γ_init']))
    elif 'ALMA' in item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_pnorm{}_iternum{}_inilr={}.pt'.format(item['model'], item['attack'],item['p_norm'], item['iter_num'],item['init_lr_dist']))
    elif 'SquareAttack' in item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_pnorm_{}_epsilon{}_iternum_{}.pt'.format(item['model'], item['attack'], item['p_norm'], item['eps'], item['steps']))
    elif 'ZOO' == item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_targeted={}_usetanh={}_solver={}.pt'.format(item['model'], item['attack'],item['targeted'], item['use_tanh'], item['solver']))

