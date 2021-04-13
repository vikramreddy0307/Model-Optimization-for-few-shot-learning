
from __future__ import division, print_function, absolute_import

import pdb
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np

class Learner(nn.Module):
    def __init__(self,image_size,batchnorm_eps,batchnorm_momentum,
                num_classes):
        super(Learner,self).__init__()
        self.clr=image_size // 2**4
        self.model=nn.ModuleDict({'features': nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, padding=1)),
            ('norm1', nn.BatchNorm2d(32, batchnorm_eps, batchnorm_momentum)),
            ('relu1', nn.ReLU(inplace=False)),
            ('pool1', nn.MaxPool2d(2)),

            ('conv2', nn.Conv2d(32, 32, 3, padding=1)),
            ('norm2', nn.BatchNorm2d(32, batchnorm_eps, batchnorm_momentum)),
            ('relu2', nn.ReLU(inplace=False)),
            ('pool2', nn.MaxPool2d(2)),

            ('conv3', nn.Conv2d(32, 32, 3, padding=1)),
            ('norm3', nn.BatchNorm2d(32, batchnorm_eps, batchnorm_momentum)),
            ('relu3', nn.ReLU(inplace=False)),
            ('pool3', nn.MaxPool2d(2)),

            ('conv4', nn.Conv2d(32, 32, 3, padding=1)),
            ('norm4', nn.BatchNorm2d(32, batchnorm_eps, batchnorm_momentum)),
            ('relu4', nn.ReLU(inplace=False)),
            ('pool4', nn.MaxPool2d(2)),
            ]))
                                  
        })
        self.model.update({'final_linear'nn.Linear(32 * clr_in * clr_in, n_classes)})
        self.criterion=nn.CrossEntropyLoss()
        
        '''
        The network then has a final linear layer followed by a softmax for the number
of classes being considered
        '''
        
    def forward(self,x):
        outputs=self.model.features(x)
        outputs = torch.reshape(outputs, [x.size(0), -1])
        outputs = self.model.cls(outputs)
        return outputs
    def flat_parmaeters(self):
        return torch.cat([p.view(-1) for p in self.model.parmeters()],0)
    def transfer_params(self, learner_w_grad, cI):
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    def reset_batch_stats(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

'''

create Meta-Learner 

'''
class Modified_MetaLSTM(nn.module):
    """C_t = f_t * C_{t-1} + i_t * (`C_t) """
    
    def __init__(self,input_size,hidden_size,n_learner_params):
        super(Modified_MetaLSTM,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.n_learner_parameters=n_learner_parameters
        
        self.weights_F=nn.Parameter(torch.tensor(input_size+2,hidden_size))
        self.weights_I=nn.Parameter(torch.tensor(input_size+2,hidden_size))

        self.bias_F=nn.Parameter(torch.Tensor(1,hidden_size))
        self.bias_I=nn.Parameter(torch.Tensor(1,hidden_size))
        
        '''
        We set the cell state of the LSTM to be the parameters of the
        learner, or ct = θt
        
        '''
        self.CellState=nn.Parameters(torch.Tensor(1,hidden_size))
        self.reset_parameters()
        
    def reset_parameters(self):
        for weight in self.parameters():
            nn.init.uniform_(weight,-0.01,0.01)
    '''
    forget gate bis to be large
    forget gate value close to 1
    input gate bias to be small
    
    '''
        nn.init.uniform_(self.bias_F,4,6)
        nn.init.uniform_(self,bias_I,-3,-1)
        
    def initialize_CellState(self,flat_params):
        self.cellState_initial.data.copy_(flat_params.unsqueeze(1))
        
    def forward(self,inputs,gates_outputs=None):
        '''
        X=Input for the current time step
        grad=
        '''
        X,grad=inputs
        batch,_=x.size()
        if gates_outputs=None:
            prev_F=torch.zeros((batch,self,hidden_size)).to(self.weights_F.device)
            prev_I=torch.zeros((batch,self.hidden_size)).to(self.weights_I.device)
            prev_C=self.cellState_initial
            hx=[prev_F,prev_I,prev_C]
        
            
        
        #Equations when gradient descent compared to LSTM gate updates
        #check the paper for further clarification
        next_F=torch.sigmoid(torch.mm(torch.cat((X,prev_C,prev_F),1),self.weights_F)+self.bias_F.expand_as(prev_F))
        next_I=torch.sigmoid(torch.mm(torch.cat((X,prev_C,prev_I),1),self.weights_I)+self.bias_I.expands_As(prev_I))
        next_C=next_F.mul(prev_C)-next_I.mul(grad)
        
        return next_C,[next_F,next_I,next_C]
    


class MetaLearner(nn.Module):
    def__init__(self,input_size,hidden_size,n_learner_params):
        super(MetaLSTM,self).__init__()
        '''
        hidden_size – The number of features in the hidden state h
        '''
        self.lstm=nn.LSTMCell(input_size=input_size,hidden_size=hidden_size)
        self.metaLSTM=MetaLSTM(input_size=input_size,hidden_size=)
    def forward(self,inputs,hidden_state=None):
        loss,preprocess_grad,grad=inputs
        loss=loss.expand_as(preprocess_grad)
        inputs=torch.cat((loss,preprocess_grad),1)
        if hidden_state is None:
            hidden_state=[None,None]
        lstm_hidden_state,lstm_cell_state=self.lstm(inputs,hidden_state[0])
        flat_learner_unsqueezed,metaLSTM_hidden_State=self.metaLSTM([lstm_hidden_state,grad],hs[1])
        return flat_learner_unsqueezed.squeeze(),[(lstm_hidden_state,lstm_cell_state),metaLSTM_hidden_State]
        
        