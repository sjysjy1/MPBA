import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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

#import sys
#sys.path.append('./sparse-rs')
from my_gene import MPBA, MPBA_adapt

from robustbench import  load_model

def seed_torch(seed=1234):
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

test_dataset=datasets.CIFAR10(root='./data',train=False,download=True,transform=transforms.ToTensor())
criterion=nn.CrossEntropyLoss()
batch_size=128
#batch_size=12

list_para=[

{'model':'WongLinf','attack':'MPBA','p_norm':'L1','popsize':32,'init_pop_range':1.0,'generation_num':2000,'stepsize_start':0.2,'stepsize_end':0.01,'batch_size':batch_size,'mu_start':1.,'mu_end':1000.,'a':None,'lower':0.1,'upper':1.0,'rate_mutation':0.1,'rate_crossover':0.,'c':1},

{'model':'Standard','attack':'MPBA','p_norm':'L1','popsize':32,'init_pop_range':1.0,'generation_num':1000,'stepsize_start':0.2,'stepsize_end':0.01,'a':None,'batch_size':batch_size,'mu_start':0.01,'mu_end':10.0,'lower':0.1,'upper':10.0,'rate_mutation':0.1,'rate_crossover':0,'c':1}, #
{'model':'Standard','attack':'MPBA','p_norm':'L1','popsize':32,'init_pop_range':1.0,'generation_num':2000,'stepsize_start':0.2,'stepsize_end':0.01,'a':None,'batch_size':batch_size,'mu_start':0.01,'mu_end':10.0,'lower':0.1,'upper':10.0,'rate_mutation':0.1,'rate_crossover':0,'c':1}, #
{'model':'Standard','attack':'MPBA','p_norm':'L1','popsize':32,'init_pop_range':1.0,'generation_num':3000,'stepsize_start':0.2,'stepsize_end':0.01,'a':None,'batch_size':batch_size,'mu_start':0.01,'mu_end':10.0,'lower':0.1,'upper':10.0,'rate_mutation':0.1,'rate_crossover':0,'c':1}, #

{'model': 'Standard','attack':'EAD','batch_size':batch_size ,'iter_num':1000},
{'model': 'WongLinf','attack':'EAD','batch_size':batch_size ,'iter_num':1000},

{'model':'Standard','attack':'FMN','p_norm':1,'steps':1000,'batch_size':batch_size,'γ_init':0.3,'α_init':0.1},
{'model':'WongLinf','attack':'FMN','p_norm':1,'steps':1000,'batch_size':batch_size,'γ_init':0.3,'α_init':0.1},

{'model': 'Standard','attack':'ALMA','p_norm':'l1','batch_size':batch_size ,'iter_num':1000,'init_lr_dist':0.5},
{'model': 'WongLinf','attack':'ALMA','p_norm':'l1','batch_size':batch_size ,'iter_num':1000,'init_lr_dist':0.5},



{'model':'WongLinf','attack':'MPBA_ada_step','p_norm':'L2','popsize':32,'init_pop_range':1.0,'generation_num':3000,'stepsize_start':0.1, 'batch_size':batch_size,'mu_start':0.01,'mu_end':1.,'lower':1.0,'upper':1.,'rate_mutation':0.1,'rate_crossover':0.,'c':1},
{'model':'WongLinf','attack':'MPBA_ada_step','p_norm':'L2','popsize':32,'init_pop_range':1.0,'generation_num':4000,'stepsize_start':0.1, 'batch_size':batch_size,'mu_start':0.01,'mu_end':1.,'lower':1.0,'upper':1.,'rate_mutation':0.1,'rate_crossover':0.,'c':1},

{'model':'Standard','attack':'MPBA_ada_step','p_norm':'L2','popsize':64,'init_pop_range':1,'generation_num':2000,'stepsize_start':0.1,'batch_size':batch_size,'mu_start':0.0,'mu_end':0.1,'lower':0.1,'upper':1.0,'rate_mutation':0.1,'rate_crossover':0,'c':1},#
{'model':'Standard','attack':'MPBA_ada_step','p_norm':'L2','popsize':64,'init_pop_range':1,'generation_num':3000,'stepsize_start':0.1,'batch_size':batch_size,'mu_start':0.0,'mu_end':0.1,'lower':0.1,'upper':1.0,'rate_mutation':0.1,'rate_crossover':0,'c':1},#

{'model':'Standard','attack':'SquareAttack','p_norm':'L2','eps':0.5,'steps':40000,'batch_size':batch_size},
{'model':'WongLinf','attack':'SquareAttack','p_norm':'L2','eps':1,'steps':40000,'batch_size':batch_size},  #run successfully for WongLinf of CIFAR10
{'model':'WongLinf','attack':'SquareAttack','p_norm':'L2','eps':2,'steps':40000,'batch_size':batch_size},
{'model':'WongLinf','attack':'SquareAttack','p_norm':'L2','eps':3,'steps':40000,'batch_size':batch_size},

{'model':'WongLinf','attack':'ZOO','solver':'adam','targeted':False,'use_tanh':True},


    ]

for item in list_para:
    list_success_fail=[]
    list_pert=[]
    list_iterNum=[]
    print(item)
    test_data = torch.tensor(test_dataset.data, device=device)
    test_labels = torch.tensor(test_dataset.targets, device=device)
    test_data_normalized = test_data / 255.0
    #test_data_normalized = test_data_normalized
    test_data_normalized = test_data_normalized.permute(0, 3, 1, 2)
    verbose=False
    if verbose:
       empty = []
       torch.save(empty, './curvedata_MNIST_{}_{}.pt'.format(item['model'],item['attack']))
    if item['model'] == 'Standard':
        model = load_model(model_name='Standard', dataset='cifar10', norm='Linf')
    elif item['model'] == 'WongLinf':
        model = load_model(model_name='Wong2020Fast', norm='Linf')
    elif item['model'] == 'WangL2WRN-28-10':
        model = load_model(model_name='Wang2023Better_WRN-28-10', dataset='cifar10', norm='L2')
    model.to(device)
    model.eval()  # turn off the dropout
    test_accuracy = False
    if test_accuracy == True:
        predict_result = torch.tensor([], device=device)
        for i in range(500):
            outputs = model(test_data_normalized[20 * i:20 * i + 20])
            _, labels_predict = torch.max(outputs, 1)
            predict_result = torch.cat((predict_result, labels_predict), dim=0)
        correct = torch.eq(predict_result, test_labels)
        torch.save(correct, './result/cifar10-first1000/{}_Cifar_correct_predict.pt'.format(item['model']))
    else:
        correct = torch.load('./result/cifar10-first1000/{}_Cifar_correct_predict.pt'.format(item['model']))
    correct_sum = correct.sum()
    clean_accuracy = correct_sum / 10000.0
    print('model clean accuracy:', clean_accuracy)
    if item['model']=='Standard':
        N=100
        correct_sum = correct[0:100].sum()
    else:
        N=1000
        correct_sum = correct[0:1000].sum()
    correct_index=[]
    for i in range(N):
        if correct[i]:
            correct_index.append(i)
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
                print('batch average of perturbation is:', perturbation.sum() / len(perturbation))
                print('success rate is:', sum(list_success_fail) / len(list_success_fail))
                end_time = time.time()
                print('time used is:', end_time - start_time)

    elif item['attack'] == 'MPBA_ada_step':
        with torch.no_grad():
             for i in range(0,len(correct_index),item['batch_size']):
                print('***************{}th batch***********'.format(int(i/item['batch_size'])))

                images=test_data_normalized[correct_index[i:i+item['batch_size']]].clone().detach()
                labels=test_labels[correct_index[i:i+item['batch_size']]]

                #outputs = model(images)
                #_, labels_predict = torch.max(outputs, 1)
                #imshow(torchvision.utils.make_grid(images[1].cpu().data, normalize=True), 'Predict:{}'.format(labels_predict[0].item()))

                adv_images=MPBA_adapt(model,images,labels,population_size=item['popsize'],init_pop_range=item['init_pop_range'],generation_num=item['generation_num'],p_norm=item['p_norm'],stepsize_start=item['stepsize_start'],mu_start=item['mu_start'],mu_end=item['mu_end'],lower=item['lower'],upper=item['upper'],rate_mutation_ini=item['rate_mutation'],rate_crossover_ini=item['rate_crossover'],c=item['c'],msg='curvedata_MNIST_'+item['model']+'_'+item['attack'])
                outs = model(adv_images)
                _, labels_predict = torch.max(outs, 1)
                success = (labels_predict != labels)
                list_success_fail = list_success_fail + success.tolist()

                #imshow(torchvision.utils.make_grid(adv_images[1].cpu().data, normalize=True), 'Predict:{}'.format(labels_predict[0].item()))

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
                print('batch average of perturbation is:', perturbation.sum() / len(perturbation))
                print('success rate is:', sum(list_success_fail) / len(list_success_fail))
                end_time = time.time()
                print('time used is:', end_time - start_time)

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
             method = partial(fmn, norm=item['p_norm'], steps=item['steps'], γ_init=item['γ_init'],
                              α_init=item['α_init'])
             attack_data = run_attack(model=model, inputs=test_data_normalized[correct_index].cpu(),
                                      labels=test_labels[correct_index].cpu(), attack=method,
                                      batch_size=item['batch_size'])

             labels_predict = torch.tensor([], device=device)
             for i in range(0, correct_sum, 16):  # evaluate 16 images once
                 outputs = model(attack_data['adv_inputs'][i:i + 16].to(device))
                 labels_predict = torch.cat((labels_predict, torch.max(outputs, 1)[1]), dim=0)
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

        labels_predict = torch.tensor([], device=device)
        for i in range(0, correct_sum, 16):  # evaluate 16 images once
            outputs = model(attack_data['adv_inputs'][i:i + 16].to(device))
            labels_predict = torch.cat((labels_predict, torch.max(outputs, 1)[1]), dim=0)
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
            #for i in range(10):
            for i in range(len(correct_index)):
                 print('i is:',i)
                 inputs,targets=zoo_l2_attack_black.generate_data(test_loader_subset,item['targeted'],samples=1,start=i-1)

                 adv_images=torch.tensor(zoo_l2_attack_black.attack(inputs-0.5, targets, model, item['targeted'], use_log, item['use_tanh'], item['solver'], device),device=device)
                 adv_images = adv_images + 0.5

                 outs = model(adv_images)
                 _, labels_predict = torch.max(outs, 1)
                 #imshow(torchvision.utils.make_grid(adv_images[0].cpu().data, normalize=True),'Predict:{}'.format(labels_predict[0].item()))
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
    print('total avg_pert is',avg_pert)
    median_pert = torch.median(torch.tensor(list_pert_success))
    print('total median value is', median_pert)
    dict_save = {'device': device, 'para': item, 'time_used': time_used, 'list_success_fail': list_success_fail,'attack_success_rate': attack_success_rate, 'list_pert': list_pert, 'avg_pert': avg_pert,'median_pert': median_pert}

    if 'MPBA' == item['attack']:
        if item['model']=='Standard':
            torch.save(dict_save,'./result/cifar10-first100/{}_attack_{}_pnorm_{}_GenNum{}_stepsize{}-{}_a={}_mu_{}-{}_lower{}_upper{}_c={}.pt'.format(item['model'],item['attack'],item['p_norm'],item['generation_num'],item['stepsize_start'],item['stepsize_end'],item['a'],item['mu_start'],item['mu_end'],item['lower'],item['upper'],item['c']))
        else:
            torch.save(dict_save,'./result/cifar10-first1000/{}_attack_{}_pnorm_{}_GenNum{}_stepsize{}-{}_a={}_mu_{}-{}_lower{}_upper{}_c={}.pt'.format(item['model'],item['attack'],item['p_norm'],item['generation_num'],item['stepsize_start'],item['stepsize_end'],item['a'],item['mu_start'],item['mu_end'],item['lower'],item['upper'],item['c']))
    elif 'MPBA_ada_step'== item['attack']:
        if item['model']=='Standard':
            torch.save(dict_save,'./result/cifar10-first100/{}_attack_{}_pnorm_{}_GenNum{}_stepsize{}_mu_{}-{}_lower{}_upper{}_c={}.pt'.format(item['model'],item['attack'],item['p_norm'],item['generation_num'],item['stepsize_start'],item['mu_start'],item['mu_end'],item['lower'],item['upper'],item['c']))
        else:
            torch.save(dict_save,'./result/cifar10-first1000/{}_attack_{}_pnorm_{}_GenNum{}_stepsize{}_mu_{}-{}_lower{}_upper{}_c={}.pt'.format(item['model'],item['attack'],item['p_norm'],item['generation_num'],item['stepsize_start'],item['mu_start'],item['mu_end'],item['lower'],item['upper'],item['c']))

    elif 'EAD' == item['attack']:
        if item['model'] == 'Standard':
            torch.save(dict_save,'./result/cifar10-first100/{}_attack_{}_iternum{}.pt'.format(item['model'], item['attack'], item['iter_num']))
        else:
            torch.save(dict_save,'./result/cifar10-first1000/{}_attack_{}_iternum{}.pt'.format(item['model'], item['attack'],item['iter_num']))
    elif 'SquareAttack' in item['attack']:
        torch.save(dict_save,'./result/cifar10-first1000/{}_attack_{}_pnorm_{}_epsilon{}_iternum_{}.pt'.format(item['model'], item['attack'], item['p_norm'], item['eps'], item['steps']))
    elif 'FMN' in item['attack']:
        if item['model'] == 'Standard':
           torch.save(dict_save,'./result/cifar10-first100/{}_attack_{}_pnorm{}_steps{}_gammaini{}.pt'.format(item['model'], item['attack'],item['p_norm'], item['steps'],item['γ_init']))
        else:
           torch.save(dict_save,'./result/cifar10-first1000/{}_attack_{}_pnorm{}_steps{}_gammaini{}.pt'.format(item['model'], item['attack'],item['p_norm'], item['steps'],item['γ_init']))
    elif 'ALMA' in item['attack']:
        if item['model'] == 'Standard':
            torch.save(dict_save,'./result/cifar10-first100/{}_attack_{}_pnorm{}_iternum{}_inilr={}.pt'.format(item['model'], item['attack'],item['p_norm'], item['iter_num'],item['init_lr_dist']))
        else:
            torch.save(dict_save,'./result/cifar10-first1000/{}_attack_{}_pnorm{}_iternum{}_inilr={}.pt'.format(item['model'], item['attack'],item['p_norm'], item['iter_num'],item['init_lr_dist']))
    elif 'ZOO' == item['attack']:
        if item['model'] == 'Standard':
            torch.save(dict_save,'./result/cifar10-first100/{}_attack_{}_targeted={}_usetanh={}_solver={}.pt'.format(item['model'], item['attack'],item['targeted'], item['use_tanh'], item['solver']))
        else:
            torch.save(dict_save,'./result/cifar10-first1000/{}_attack_{}_targeted={}_usetanh={}_solver={}.pt'.format(item['model'], item['attack'],item['targeted'], item['use_tanh'], item['solver']))


