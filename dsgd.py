# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:31:06 2019

@author: Debjyoti Bhattacharjee
Reference implementation of DSSGD based on the paper
'Privacy-Preserving Deep Learning' by Shokri et al.
"""
import copy
from heapq import *
import math
import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))        
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def updateFromServer (network_dict, old_net_dict):
    global global_net_dict, global_update_dict, params_updated
    global_net_dict = copy.deepcopy(network_dict)
    
   
    #get the top most values along with the delta
    heap = []
    for key, val in network_dict.items():
        addHeap(val,heap, params_updated, key,[],old_net_dict[key])
    
    # update the global dict
    for val, key, index in heap:
        if len(index) == 1: # just an array
            global_net_dict[key][index[0]] = global_net_dict[key][index[0]]  + val
            global_update_dict[key][index[0]] = global_update_dict[key][index[0]]  + 1
        else:
            gval = global_net_dict[key][index[0]]
            g_upval = global_update_dict[key][index[0]] 
            index = index[1:]
            while len(index) > 1:
                gval = gval[index]
                g_upval = g_upval[index]
                index = index[1:]
            gval[index[0]] = gval[index[0]] + val
            g_upval[index[0]] = g_upval[index[0]] + 1
    heap = []
    for key, val in global_update_dict.items():
        addHeap(val,heap, params_updated, key,[])
    #print(heap)
    
    #create the download list for the most updated parameters
    #hack : update the network dict directly with the parameters!!!
    for val, key, index in heap:
        if len(index) == 1: # just an array
            network_dict[key][index[0]] = global_net_dict[key][index[0]]
            
        else:
            gval = global_net_dict[key][index[0]]
            nval = network_dict[key][index[0]]
            index = index[1:]
            while len(index) > 1:
                gval = gval[index]
                nval = nval[index]
                index = index[1:]
            nval[index[0]] = gval[index[0]] 
            
    return network_dict

def train(network,optimizer, train_losses, train_counter, train_loader, epoch, batch_idx_start=0, batch_idx_complete=None, fname_suffix=''):
  network.train()
  old_net_dict = network.state_dict()
  for batch_idx, (data, target) in enumerate(train_loader):
    # train only a specific range of batches
    
    if batch_idx_start > batch_idx:
        continue
    if batch_idx_complete != None and batch_idx_complete < batch_idx:
        break
    
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} {}[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), './results/model'+fname_suffix+'.pth')
      torch.save(optimizer.state_dict(), './results/optimizer'+fname_suffix+'.pth')

  # upload a fraction of the current weights and update the current network with
  # provided weights
  state_dict = network.state_dict()
  state_dict = updateFromServer(state_dict,  old_net_dict)  
  network.load_state_dict(state_dict)
  print(id(state_dict), id(network.state_dict()))
  torch.save(network.state_dict(), './results/model'+fname_suffix+'.pth')
  torch.save(optimizer.state_dict(), './results/optimizer'+fname_suffix+'.pth')
  return network,optimizer

      
      

def test(network,test_losses, test_loader):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
 

# build the actual networks
def runDSSGD(trainers, n_epochs, part_size, n_parts):
    global global_update_dict
    network = list()
    optimizer = list()
    train_losses = list()
    train_counter = list()
    test_losses = list()
    test_counter = list()
    weight = list()
    
    for i in range(trainers):
        network.append(Net())
        optimizer.append(optim.SGD(network[i].parameters(), lr=learning_rate, momentum=momentum))
        train_losses.append([])
        train_counter.append([])
        test_losses.append([])
        test_counter.append([j*len(train_loader) for j in range(n_epochs + 1)])
        weight.append(None)
        
    for epoch in range(1, n_epochs+1):
        for part in range(n_parts):
            t_list = [i for i in range(trainers)]
            random.shuffle(t_list)
            batch_idx_start = part_size*part
            batch_idx_complete = part_size*(part+1)  # split each epoch
            print('Training order:',t_list)
            for trainer in t_list:
                print('trainer: %d epoch: %d batch start: %d end : %d' % \
                      (trainer,epoch,batch_idx_start,batch_idx_complete))
                network[trainer],optimizer[trainer] = train(network[trainer],\
                      optimizer[trainer],\
                      train_losses[trainer],\
                      train_counter[trainer],\
                      train_loader_part[trainer],\
                      epoch,\
                      batch_idx_start,\
                      batch_idx_complete,\
                      '_t'+str(trainer))
                
        for trainer in range(trainers):
            print("Trainer:%d Test:%d" % (trainer,epoch))
            test(network[trainer],\
                 test_losses[trainer],\
                 test_loader)
    #print(global_update_dict)
            
            
def addHeap(val, heap, k, key, index=[], old_val = None):
    if(len(list(val.shape)) == 1): # inner most array
        for i in range(len(val)):
            #if len(heap) > 0:
                #print(val[i], heap[0], len(heap))
            if len(heap) >= k and val[i] > heap[0][0]:
                heappop(heap)
                if type(old_val) != type(None):
                    heappush(heap, (val[i]-old_val[i], key, index+[i]))
                else:
                    heappush(heap, (val[i], key, index+[i]))
            elif len(heap) < k :
                if type(old_val) != type(None):
                    heappush(heap, (val[i]-old_val[i], key, index+[i]))
                else:
                    heappush(heap, (val[i], key, index+[i]))
    else:

        for i in range(len(val)):
            if type(old_val) != type(None):
                addHeap(val[i],heap,k,key,index+[i], old_val[i])
            else:
                addHeap(val[i],heap,k,key,index+[i])
                
def getInfo(net_dict, theta=1):
    net_dict_info  = []
    heap_delta = []
    total_param = 0
    for key,val in net_dict.items():
        print(key, val.shape)
        v_flat = 1
        for dim in val.shape:
            v_flat = v_flat*dim 
        #print(key, v_flat, list(val.shape))
        total_param = total_param + v_flat
        
        net_dict_info.append([key,val.shape,v_flat])
        '''
        top_k = 3
        l_val = largest_indices(val, top_k)
        print(l_val)
        v = None 
        for k in range(top_k):
            v = val
            for l in l_val:
                index = l[k]
                v = v[index]
            print(v) '''
    params_updated = math.floor(theta*total_param)
    print('Total param: %d Theta: %.2f Updated: %d' % (total_param, theta, params_updated))
    return total_param, params_updated, net_dict_info
    #for key,val in net_dict.items():
    #    addHeap(val,heap_delta, paramsUpdated, key)
    #print('largest elements:',heap_delta)
    #print(net_dict_info)
        

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)
    
trainers = 20 # number of participants in DSSGD
partition_ratio = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,10,10,1,1]
total = sum(partition_ratio)
partition_ratio = [i*1.0/total for i in partition_ratio]
print(partition_ratio)
n_parts = 2 # number of times global update happens per epoch

theta = 0.4
n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.manual_seed(random_seed)
torch.backends.cudnn.enabled = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
global_net = Net()
global_net_dict = global_net.state_dict()
print(global_net_dict.keys())

global_update_dict = dict()
#print(global_net_dict)
total_param, params_updated, net_dict_info = getInfo(global_net_dict,theta)
for key, shape,v_flat in net_dict_info:
    global_update_dict[key] = np.zeros(shape)
    print(global_update_dict[key].shape)
train_loader = torchvision.datasets.MNIST('./datafiles/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
                              
# partition the data between the trainers
train_samples = len(train_loader)
print('Total samples %d' % (train_samples)) 
partition_index = [list() for i in range(trainers)]
indices = [i for i in range(train_samples)]
random.shuffle(indices)
partition_len = math.floor(train_samples/trainers)
train_loader_part = []
index = 0
for i in range(trainers):
    if index + math.floor(partition_ratio[i]*train_samples) < train_samples:
        lim = index + int(math.floor(partition_ratio[i]*train_samples))
    else:
        lim = train_samples
    partition_index[i]= indices[index:lim]
    index = lim
    dataset = []
    for j in partition_index[i]:
        dataset.append(train_loader[j])
    print('parition index:', len(partition_index[i]))
    train_loader_part.append(torch.utils.data.DataLoader(dataset,\
                                                         batch_size=batch_size_train,\
                                                         shuffle=True))
    print('dataset length [%d]: %d ' %(i, len(train_loader_part[i].dataset)))

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./datafiles/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)
                                                             
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)      
print(example_data.shape) 



runDSSGD(trainers,\
        n_epochs,\
        math.floor(len(train_loader_part[i].dataset)/(batch_size_train*n_parts)),\
        n_parts)

'''
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)  
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader) for i in range(n_epochs + 1)]    

test(network,test_loader)
for epoch in range(1, n_epochs + 1):
  print(len(train_loader_part[i].dataset))
  if epoch > 1:
      batch_idx_complete = 150
  else:
      batch_idx_complete = 0
  train(network,train_loader_part[i],epoch,batch_idx_complete)
  test(network,test_loader)     '''                     
