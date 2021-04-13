from __future__ import division, print_function, absolute_import
import os
import copy
import random
import argparse
import torch
import torch.nn as nn
import numpy as np
from learner import Learner
from metalearner import MetaLearner
from dataloader import prepare_data


def main():
    training_data,validation_data,test_data=prepare_data()
    
    learner=Learner(image_size,eps_BatchNorm,momentum_BatchMomentum)
    '''
    input_size="Input size for the first LSTM"
    hiden_size="Hidden size for the first LSTM"
    '''
    metalearner=MetaLeatner(input_size,hidden_size,learner.get_flat_parameters().size(0))
    metalearner.metalearner_initialize_cellstate(learner.get_flat_params())
    
    
     # set up loss , optimizer
    optim =torch.optim.Adma(metalearner.parameters(),args.learning_rate)
    
    
    
    best_acc=0
    for eps, (episode_X,episode_Y) in enumerate(train_loader):
        '''
        episode_x.shape = [n_class, n_shot + n_eval, c, h, w]
        '''
        
        train_X = episode_X[:, :args.n_shot].reshape(-1, *episode_x.shape[-3:])#[num_class * n_shot,:]
        train_Y=torch.LongTensor(np.repeat(range(args.num_classes),args.n_shot))#[num_class * n_shot]
        test_X=episode_X[:,args.n_shot:].reshape(-1,*episode_x.shape[-3:])# [num_class * n_eval, :]
        test_Y=torch.LongTensor(np.repeat(range(args.num_classes),args.n_eval))#[num_class * n_eval]
        
        #train learner and meta learner
        learner.reset_batch_stats()
        learner_for_testing.reset_batch_stats()
        learner_for_testing.train()
        learner.train()
        metalearner_cell_state=metalearner.metalstm.cI.data
        #cI = train_learner(learner_w_grad, metalearner, train_input, train_target, args)
        for epoch in range(args.epoch):
            for i in rage(0,len(train_X),args.batch_size):
                train_X_batch=train_X[i:i+args.batch_size]
                train_Y_batch=train_Y[i:i+args.batch_size]
                
                learner.copy_flat_params(metalearner_cell_state)
                output=learner(train_X_batch)
                loss=learner.criterion(output,train_Y_batch)
                acc=accuracy(output,train_Y_batch)

                #get the loss of the learner on train batch

                learner.zero_grad()
                loss.backward()
                grad=torch.cat([p.grad.data.view(-1) / args.batch_size for p in learner_w_grad.parameters()], 0)


                '''
                3.2 PARAMETER SHARING & PREPROCESSING

                 preprocessing method of Andrychowicz et al. (2016)
                worked well when applied to both the dimensions of the gradients
                and the losses at each time step

                '''
                preprocessed_grad=preprocess_grad(grad)
                preprocessed_loss=preprocess_loss(loss.data.unsqueeze(0))

                #line 10 
                metalearner_input=[preprocessed_loss,preprocessed_grad,grad.unsqueeze(0)]
                cellstate_metalearner,hiddenstate_metalearner=metalearner([metalearner_input,hidden_state[-1]])
                #line 11
                hidden_state.append(hiddenstate_metalearner)

        '''
        refer 3.4 BATCH NORMALIZATION-- mandatory
        , we found that a better strategy was to collect statistics for each dataset D ∈ D
        during Dmeta−test, but then erase the running statistics when we consider the next dataset
        '''

        learner_for_testing.tranfer_parameters(learner,cellstate_metalearner)
        test_output=learner_for_testing(test_X)
        test_loss=learner_for_testinf(test_output,test_Y)
        test_acc=accuracy(test_output,test_Y)
        optim.zero_grad()
        test_loss.backward()
        '''
        How to fix exploding gradients: gradient clipping

        '''
        nn.utils.clip_grad_norm_(metalearner.parameters(), args.grad_clip)
        optim.step() #update parameters