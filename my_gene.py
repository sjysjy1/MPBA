import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

def sign_fun(x):
    cond=x<0
    y=torch.full_like(x,1,device=x.device)
    y[cond]=0
    return y


def fitness_fun_minpert(model,images,labels, images_ori ,mu,p_norm,targeted):
    #optimizer.zero_grad()
    if p_norm == 'L0':
        fitness1=torch.norm((images - images_ori).view(len(images), -1), p=0, dim=1)
    elif p_norm == 'L1':
        fitness1=torch.norm((images - images_ori).view(len(images), -1), p=1, dim=1)
    elif p_norm == 'L2':
        fitness1=torch.norm((images - images_ori).view(len(images), -1), p=2, dim=1)
    elif p_norm == 'Linf':
        fitness1=torch.norm((images - images_ori).view(len(images), -1), p=float('inf'), dim=1)
    outs = model(images)
    one_hot_labels = torch.eye(outs.shape[1]).to(images.device)[labels]
    other = torch.max((1 - one_hot_labels) * outs, dim=1)[0]
    real = torch.max(one_hot_labels * outs, dim=1)[0]
    if  ~targeted:
        fitness2= (real-other)*sign_fun(real-other)
    else:
        fitness2 = (other- real ) * sign_fun(real - other)
    return [(fitness1+mu*fitness2).tolist(), (real-other).tolist()]


#not using binary bit
def MPBA(model,
                    images,
                    labels,
                    p_norm:str='L2',
                    population_size:int=16,
                    init_pop_range:float=1.0,
                    tounament_size:int=3,
                    rate_mutation_ini:float=0.1,
                    stepsize_start:float=0.2,
                    stepsize_end:float=0.01,
                    a:float=1,
                    rate_crossover_ini=0.5,
                    mu_start:float=0.01,#penalty parameter
                    mu_end:float=100,
                    c:float=1,
                    alpha:float=0.5, #keep rate
                    generation_num:int=2000,#:int=10000,
                    targeted:bool=False,
                    verbose:bool=False,
                    #adaptive_mu:bool=True,
                    lower:float=1.0,
                    upper:float=10.0,
                    msg:str=None #for plotting

):
    with torch.no_grad():
        assert not population_size is None
        assert p_norm in ['L0','L1','L2','Linf',]
        device=images.device
        images_ori=images.clone().detach().to(device)
        best_adv = images.clone().detach().to(device)
        best_adv_norm = torch.full((len(images),), float('inf'), device=device)
        population=torch.tensor([],device=device)
        for i in range(population_size):
            chromo = torch.clamp(images + (2*torch.rand( size=images.shape, device=device)-1)*init_pop_range,min=0,max=1)
            population=torch.cat((population,torch.unsqueeze(chromo,dim=1)),dim=1)

        cnt=0
        best_iter=[]
        mu = (mu_end-mu_start) * (cnt / generation_num) ** c+mu_start
        predict_result = torch.tensor([fitness_fun_minpert(model, population[:, i, :, :, :], labels, images_ori, mu,p_norm=p_norm, targeted=targeted) for i in range(population_size)],device=device)
        scores=predict_result[:,0,:].T
        outs_diff=predict_result[:,1,:].T
        scores_sorted, idx_sorted = torch.sort(scores, dim=1)

        k = 5
        mask_feasible_fifo=torch.tensor([True, False],device=device)
        mask_feasible_fifo=mask_feasible_fifo.repeat(len(images),k)
        mu_end_min = lower*mu_end
        mu_end_max = upper * mu_end
        mu_start=torch.full(size=(len(images),),fill_value=mu_start,device=device)
        mu_end = torch.full(size=(len(images),),fill_value=mu_end,device=device)

        while cnt<generation_num:
            if a==None:
               sigma = 1/2*(stepsize_start - stepsize_end) * (1 + math.cos(cnt*math.pi / generation_num)) + stepsize_end
            else:
               sigma = (stepsize_start - stepsize_end) * (1 - cnt / generation_num) ** a + stepsize_end
            #print('cnt={},sigma={:4f},best score is:{}'.format(cnt, sigma, scores_sorted[:, 0]))

            rate_mutation = rate_mutation_ini * (1 - 0.9 * cnt / generation_num)
            rate_crossover= rate_crossover_ini
            if verbose==True:
                best_iter.append(scores_sorted[:,0].mean().item())

            parent_idx=torch.tensor([],device=device,dtype=torch.int64)
            for i in range(population_size):
                idx=torch.randint(low=0, high=population_size, size=(len(population),tounament_size)).to(device)
                tournament=torch.gather(scores, dim=1, index=idx)
                argidx_idx=torch.argmin(tournament,dim=1)
                parent_idx=torch.cat((parent_idx,torch.gather(idx,dim=1,index=torch.unsqueeze(argidx_idx,dim=1))),dim=1)

            # crossover on two parents to generate two children
            children=torch.tensor([],device=device)
            for i in range(0,population_size,2):
                parent1,parent2=population[range(len(population)),parent_idx[:, i]],population[range(len(population)),parent_idx[:, i+1]]
                crossover_mask=torch.rand(size=parent1.shape,device=device,dtype=torch.float32) < rate_crossover
                extrapolate_c=1.2
                beta=torch.rand(size=parent1.shape,dtype=torch.float32,device=device)*extrapolate_c
                parent1_new=parent1-beta*(parent1-parent2)
                parent2_new = parent2 + beta * (parent1 - parent2)
                child1=torch.where(crossover_mask,parent1,parent1_new)
                child2=torch.where(crossover_mask,parent2,parent2_new)
                children=torch.cat((children,torch.unsqueeze(child1,dim=1),torch.unsqueeze(child2,dim=1)),dim=1)
            #mutation for all children
            mutation_mask=(torch.rand(size=population.shape,device=device)<rate_mutation)
            if  p_norm=='L0' or p_norm=='L1' or p_norm=='L2' :
               mutation_noise=torch.normal(mean=0,std=sigma,size=population.shape,device=device)
            elif p_norm=='Linf' :
               mutation_noise=0.01*torch.rand(size=population.shape,device=device)

            children = torch.clamp(children + mutation_mask * mutation_noise, min=0, max=1.0)
            if (p_norm == 'L0' or p_norm == 'L1') :
                for i in range(population_size):
                    (children[:,i,:,:,:])[torch.abs(children[:,i,:,:,:]-images_ori)<sigma]=images_ori[torch.abs(children[:,i,:,:,:]-images_ori)<sigma]
            #keep the top choromosome
            children_predict_result=torch.tensor([fitness_fun_minpert(model,children[:,i,:,:,:],labels,images_ori,mu,p_norm=p_norm,targeted=targeted) for i in range(population_size)],device=device)
            children_scores=children_predict_result[:, 0, :].T
            children_outs_diff = children_predict_result[:, 1, :].T
            children_scores_sorted, children_idx_sorted = torch.sort(children_scores, dim=1)
            torch.set_printoptions(precision=4)

            N_keep=int(alpha*population_size)
            N_new = int((1-alpha) * population_size)
            population = torch.cat((population[torch.arange(population.size(0)).unsqueeze(1), idx_sorted[:, 0:N_keep]], children[torch.arange(population.size(0)).unsqueeze(1),children_idx_sorted[:, 0:N_new]]),dim=1)
            scores=torch.cat((scores[torch.arange(population.size(0)).unsqueeze(1), idx_sorted[:, 0:N_keep]], children_scores[torch.arange(population.size(0)).unsqueeze(1),children_idx_sorted[:, 0:N_new]]),dim=1)
            outs_diff=torch.cat((outs_diff[torch.arange(population.size(0)).unsqueeze(1), idx_sorted[:, 0:N_keep]], children_outs_diff[torch.arange(population.size(0)).unsqueeze(1),children_idx_sorted[:, 0:N_new]]),dim=1)
            scores_sorted, idx_sorted = torch.sort(scores, dim=1)
            cnt=cnt+1

            mask_feasible_fifo = torch.cat((torch.unsqueeze(outs_diff[range(len(images)), idx_sorted[:, 0]] < 0, dim=1), mask_feasible_fifo[:, :-1]),dim=1)
            mask_all_fea = torch.all(mask_feasible_fifo, dim=1)
            mask_all_infea = torch.all(~mask_feasible_fifo, dim=1).cpu()
            mu_end[mask_all_infea] = 2*mu_end[mask_all_infea]
            mu_end[mask_all_fea] = 0.5 * mu_end[mask_all_fea]
            mu_end=torch.clamp(mu_end,min=mu_end_min,max=mu_end_max)

            mu = (mu_end-mu_start) * (cnt / generation_num) ** c+mu_start


            if p_norm == 'L1':
                norm = torch.norm((population[torch.arange(population.size(0)), idx_sorted[:, 0]] - images_ori).view(len(images), -1),p=1, dim=1)
            elif p_norm == 'L2':
                norm = torch.norm((population[torch.arange(population.size(0)), idx_sorted[:, 0]] - images_ori).view(len(images), -1),p=2, dim=1)
            is_adv = (outs_diff[torch.arange(population.size(0)),idx_sorted[:,0]] < 0)
            is_better_adv = is_adv & (norm < best_adv_norm)
            best_adv = torch.where(is_better_adv[:, None, None, None],population[torch.arange(population.size(0)), idx_sorted[:,0]], best_adv)
            best_adv_norm = torch.where(is_better_adv, norm, best_adv_norm)
        #torch.save(best_iter,'./curves/best_iter.pt')
        #with open('./curvedata_best_iter_all_patch.txt','a') as f:
        #    f.write(str(best_iter))
        if verbose:
           curvedata_best_iter_all_patch=torch.load('./{}.pt'.format(msg))
           curvedata_best_iter_all_patch.append(best_iter)
           torch.save(curvedata_best_iter_all_patch,'./{}.pt'.format(msg))

        if verbose == True:  # 绘制收敛曲线
            plot_path = './'
            plt.figure(figsize=(8, 6))
            plt.tick_params(size=5, labelsize=13)  # 坐标轴
            plt.grid(alpha=0.3)  # 是否加网格线
            plt.plot(np.arange(generation_num), best_iter, color='#e74c3c', lw=1.5)
            plt.xlabel('Iteration', fontsize=13)
            plt.ylabel('Objective function value', fontsize=13)
            plt.title('Genetic Algorithm', fontsize=15)
            if plot_path is not None:
                plt.savefig(f'{plot_path}/GeneticAlgorithm_iter{generation_num}.svg', format='svg', bbox_inches='tight')
            plt.show()

        return best_adv



#adaptive mutation step size using the famous 1/5 rule
def MPBA_adapt(model,
                    images,
                    labels,
                    p_norm:str='L2',
                    population_size:int=16,
                    init_pop_range:float=1.0,
                    tounament_size:int=3,
                    #keep_size: int=10,
                    rate_mutation_ini:float=0.1,
                    #stepsize:float=0.1, #standard deviation for Gaussian mutation
                    stepsize_start:float=0.2,
                    rate_crossover_ini=0.5,
                    mu_start:float=0.01,#penalty parameter
                    mu_end:float=100,
                    c:float=1,
                    alpha:float=0.5, #keep rate
                    generation_num:int=2000,#:int=10000,
                    targeted:bool=False,
                    verbose:bool=True,
                    #adaptive_mu:bool=True,
                    lower:float=1.0,
                    upper:float=10.0,
                    msg:str=None #for plotting

):
    with torch.no_grad():
        assert not population_size is None
        assert p_norm in ['L0','L1','L2','Linf',]
        device=images.device
        images_ori=images.clone().detach().to(device)
        best_adv = images.clone().detach().to(device)
        best_adv_norm = torch.full((len(images),), float('inf'), device=device)


    #    rate_mutation = 0.01
        #init_pop_range = 1
        population=torch.tensor([],device=device)
        for i in range(population_size):
            #init_pop_range=(i+1)/population_size
            #chromo=images+torch.normal(mean=0,std=0.1,size=images.shape,device=device)
            chromo = torch.clamp(images + (2*torch.rand( size=images.shape, device=device)-1)*init_pop_range,min=0,max=1)
            population=torch.cat((population,torch.unsqueeze(chromo,dim=1)),dim=1)


    #    #population = [torch.randint(0, 2, (population_size,n_bits,)).to(device) for _ in range(images.shape[0])]
    #    population = torch.rand(size=(images.shape[0],population_size,images.shape[1],images.shape[2],images.shape[3]),dtype=torch.float32,device=device)
        cnt=0
        best_iter=[]
        #mu = (c * cnt / generation_num + 0.1)
        #mu = (mu_end - mu_start) * cnt / generation_num + mu_start
        mu = (mu_end-mu_start) * (cnt / generation_num) ** c+mu_start
        predict_result = torch.tensor([fitness_fun_minpert(model, population[:, i, :, :, :], labels, images_ori, mu,p_norm=p_norm, targeted=targeted) for i in range(population_size)],device=device)
        #scores = torch.torch.tensor([fitness_fun_minpert(model, population[:, i, :, :, :], labels, images_ori, mu,p_norm=p_norm, targeted=targeted) for i in range(population_size)], device=device).T
        scores=predict_result[:,0,:].T
        outs_diff=predict_result[:,1,:].T
        scores_sorted, idx_sorted = torch.sort(scores, dim=1)

        k = 5
        mask_feasible_fifo=torch.tensor([True, False],device=device)
        mask_feasible_fifo=mask_feasible_fifo.repeat(len(images),k)

        mu_end_min = lower*mu_end
        mu_end_max = upper * mu_end
        mu_start=torch.full(size=(len(images),),fill_value=mu_start,device=device)
        mu_end = torch.full(size=(len(images),),fill_value=mu_end,device=device)

        sigma=torch.full(size=(len(images),),fill_value=stepsize_start,device=device)

        while cnt<generation_num:
            #if (cnt+1)%100==0:
            #print('cnt={},best score is:{}'.format(cnt, scores_sorted[:, 0]))
            #print('sigma is:',sigma)

            rate_mutation = rate_mutation_ini * (1 - 0.9 * cnt / generation_num)
            #rate_mutation = rate_mutation_ini * (1 - 0.9 * cnt / generation_num)**2
            rate_crossover= rate_crossover_ini #* (1 - 0.9 * cnt / generation_num)
            if verbose==True:
                best_iter.append(scores_sorted[:,0].mean().item())

            parent_idx=torch.tensor([],device=device,dtype=torch.int64)
            for i in range(population_size):
                idx=torch.randint(low=0, high=population_size, size=(len(population),tounament_size)).to(device)
                tournament=torch.gather(scores, dim=1, index=idx)
                argidx_idx=torch.argmin(tournament,dim=1)
                parent_idx=torch.cat((parent_idx,torch.gather(idx,dim=1,index=torch.unsqueeze(argidx_idx,dim=1))),dim=1)


            # crossover on two parents to generate two children
            children=torch.tensor([],device=device)
            for i in range(0,population_size,2):
                #parent1, parent2 = torch.gather(population, parent_idx[i]), torch.gather(population,parent_idx[i + 1])
                #tmp=parent_idx[:, i]
                parent1,parent2=population[range(len(population)),parent_idx[:, i]],population[range(len(population)),parent_idx[:, i+1]]

                crossover_mask=torch.rand(size=parent1.shape,device=device,dtype=torch.float32) < rate_crossover
                extrapolate_c=1.2
                beta=torch.rand(size=parent1.shape,dtype=torch.float32,device=device)*extrapolate_c
                parent1_new=parent1-beta*(parent1-parent2)
                parent2_new = parent2 + beta * (parent1 - parent2)
                child1=torch.where(crossover_mask,parent1,parent1_new)
                child2=torch.where(crossover_mask,parent2,parent2_new)
                children=torch.cat((children,torch.unsqueeze(child1,dim=1),torch.unsqueeze(child2,dim=1)),dim=1)


            #mutation for all children

            if  p_norm=='L0' or p_norm=='L1' or p_norm=='L2' :
                for i in range(len(images)):
                    mutation_mask = (torch.rand(size=population[i].shape, device=device) < rate_mutation)
                    mutation_noise=torch.normal(mean=0,std=sigma[i],size=population[i].shape,device=device)
                    children[i] = torch.clamp(children[i] + mutation_mask * mutation_noise, min=0, max=1.0)
                    if p_norm=='L1':
                        (children[i])[torch.abs(children[i] - torch.unsqueeze(images_ori[i],dim=0).repeat(population_size,1,1,1)) < sigma[i]] = torch.unsqueeze(images_ori[i],dim=0).repeat(population_size,1,1,1)[torch.abs(children[i] - images_ori[i]) < sigma[i]]

            elif p_norm=='Linf' :
               mutation_noise=0.01*torch.rand(size=population.shape,device=device)

            #keep the top choromosome
            children_predict_result=torch.tensor([fitness_fun_minpert(model,children[:,i,:,:,:],labels,images_ori,mu,p_norm=p_norm,targeted=targeted) for i in range(population_size)],device=device)
            children_scores=children_predict_result[:, 0, :].T
            children_outs_diff = children_predict_result[:, 1, :].T
            children_scores_sorted, children_idx_sorted = torch.sort(children_scores, dim=1)
            torch.set_printoptions(precision=4)
            increase_ratio=torch.sum(children_scores < torch.gather(scores,1,parent_idx),dim=1)/population_size
            inc_sigma_idx=increase_ratio>0.2
            dec_sigma_idx=increase_ratio<0.2
            sigma[inc_sigma_idx]=sigma[inc_sigma_idx]/0.9
            sigma[dec_sigma_idx] = 0.9 * sigma[dec_sigma_idx]
            sigma=torch.clamp(sigma,min=0.001,max=0.2)



            N_keep=int(alpha*population_size)
            N_new = int((1-alpha) * population_size)
            population = torch.cat((population[torch.arange(population.size(0)).unsqueeze(1), idx_sorted[:, 0:N_keep]], children[torch.arange(population.size(0)).unsqueeze(1),children_idx_sorted[:, 0:N_new]]),dim=1)
            scores=torch.cat((scores[torch.arange(population.size(0)).unsqueeze(1), idx_sorted[:, 0:N_keep]], children_scores[torch.arange(population.size(0)).unsqueeze(1),children_idx_sorted[:, 0:N_new]]),dim=1)
            outs_diff=torch.cat((outs_diff[torch.arange(population.size(0)).unsqueeze(1), idx_sorted[:, 0:N_keep]], children_outs_diff[torch.arange(population.size(0)).unsqueeze(1),children_idx_sorted[:, 0:N_new]]),dim=1)
            scores_sorted, idx_sorted = torch.sort(scores, dim=1)
            cnt=cnt+1


            mask_feasible_fifo = torch.cat((torch.unsqueeze(outs_diff[range(len(images)), idx_sorted[:, 0]] < 0, dim=1), mask_feasible_fifo[:, :-1]),dim=1)
            mask_all_fea = torch.all(mask_feasible_fifo, dim=1)
            mask_all_infea = torch.all(~mask_feasible_fifo, dim=1).cpu()
            mu_end[mask_all_infea] = 2*mu_end[mask_all_infea]
            mu_end[mask_all_fea] = 0.5 * mu_end[mask_all_fea]
            mu_end=torch.clamp(mu_end,min=mu_end_min,max=mu_end_max)

            mu = (mu_end-mu_start) * (cnt / generation_num) ** c+mu_start

            if p_norm == 'L1':
                norm = torch.norm((population[torch.arange(population.size(0)), idx_sorted[:, 0]] - images_ori).view(len(images), -1),p=1, dim=1)
            elif p_norm == 'L2':
                norm = torch.norm((population[torch.arange(population.size(0)), idx_sorted[:, 0]] - images_ori).view(len(images), -1),p=2, dim=1)
            is_adv = (outs_diff[torch.arange(population.size(0)),idx_sorted[:,0]] < 0)
            is_better_adv = is_adv & (norm < best_adv_norm)
            best_adv = torch.where(is_better_adv[:, None, None, None],population[torch.arange(population.size(0)), idx_sorted[:,0]], best_adv)
            best_adv_norm = torch.where(is_better_adv, norm, best_adv_norm)
        torch.save(best_iter, './curves/best_iter.pt')
        return best_adv
