import sys 
import numpy as np
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
import os
import pandas as pd
from dataset import LocalAccsPlot, plotsShow, LossesPlot, skinCancer,chexpert, drich_non_skin
from models import ConvSkin
from utils import grad_vec, count_parameters, grad_dec_global, weight_vec, weight_dec_global
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
from scipy.spatial import distance_matrix, distance
from pyod.models.copod import COPOD
from skincancer_read import get_data,compute_img_mean_std, CustomDataset
import pickle as pkl
import random
import dataset
import torchxrayvision as xrv


plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 80

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class Federated:
  def __init__(self, numOfAgents,numOfClasses,data, modelDiff, B, seedo):
    self.seedo = seedo
    np.random.seed(seedo)
    torch.manual_seed(seedo)
    random.seed(seedo)
    self.numOfClasses = numOfClasses
    self.numOfAgents = numOfAgents # num of agents to be used
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(self.device)
    self.Done = False
    self.modelDiff = modelDiff
    self.data = data
    self.selected = []
    self.B = B
    if self.data == 'chexpert':
        self.rocs = [[] for i in range(numOfClasses)]
    if self.data == 'noniid_skincancer':
        self.rocs = []
    if data == 'chexpert': self.chexpert()
    if data == 'noniid_skincancer': self.noniid_skincancer()
        
  

  
    
  def noniid_skincancer(self):
        train_loader, test_loader = dataset.skinCancer_drich(self.numOfAgents)
        self.train_data, self.test_data = dataset.drich_non_skin(self.numOfAgents,train_loader),  dataset.drich_non_skin(self.numOfAgents,test_loader)

  def chexpert(self):
    self.train_data, self.test_data = chexpert(self.B)  

  def preprocess_fed(self,data):
    nonIID_tensors = [[] for i in range(self.numOfAgents)]  
    nonIID_labels = [[] for i in range(self.numOfAgents)]  

    for i in data:
        if i[1] < self.numOfClasses//2:
            j = np.random.randint(0, high  = self.numOfAgents/2)
            nonIID_tensors[j].append(i[0])
            nonIID_labels[j].append(torch.tensor(i[1]).reshape(1))

        elif i[1] >= self.numOfClasses//2: 
            j = np.random.randint(self.numOfAgents//2, high  = self.numOfAgents)
            nonIID_tensors[j].append(i[0])
            nonIID_labels[j].append(torch.tensor(i[1]).reshape(1))

    dataset = [[] for i in range(self.numOfAgents) ]
    for i in range(self.numOfAgents):
        dataset[i].append([torch.stack(nonIID_tensors[i]),torch.cat(nonIID_labels[i])])
    
    return dataset
          
  def show_distribution(self):
    sns.set_theme(style="whitegrid")
    Agents_dist = []
    npAgentsDist = np.zeros((self.numOfAgents,self.numOfClasses))
    Agents_names = ['Agent' + str(i) for i in range(self.numOfAgents)]
    for agent in range(self.numOfAgents):
        Dist = [np.count_nonzero(self.train_data[agent][0][1] == j) for j in range(self.numOfClasses)]
        npAgentsDist[agent,:] = Dist
    npAgentsDist = pd.DataFrame(npAgentsDist)
    if self.data == 'iid_skincancer':
        npAgentsDist = npAgentsDist.set_axis(['C0','C1','C2','C3','C4','C5','C6'], axis = 1, inplace=False)
    else:
        npAgentsDist = npAgentsDist.set_axis(['C0','C1','C2','C3','C4','C5','C6','C7','C8','C09'], axis = 1, inplace=False)
    npAgentsDist['Agents'] = ['A'+str(i) for i in range(self.numOfAgents)]
    g = npAgentsDist.set_index('Agents').plot(
        kind = 'bar', stacked = True, color=['skyblue', 'green','b','brown','orange','silver','navy','magenta','black','y']
    ).legend(loc='center left', bbox_to_anchor=(1, 0.5))
    

  def initilize_models(self,modelName,learning_rate, lr):
  
        
    if modelName == 'ConvSkin':
        self.Global_Model = ConvSkin().to(self.device)
        self.criterion = nn.CrossEntropyLoss() # loss function
        self.clientModels = [ copy.deepcopy(self.Global_Model) for i in range(self.numOfAgents)  ] 
        
    if modelName == 'ResNet18':
        self.Global_Model = client = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False).to(self.device)
        if self.data == 'chexpert':
            self.Global_Model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.Global_Model.fc = nn.Linear(512,13)
            self.criterion = nn.BCEWithLogitsLoss()
            self.Global_Model.cuda()
        if self.data == 'noniid_skincancer':
            self.criterion = nn.CrossEntropyLoss() # loss function
        self.clientModels = [ copy.deepcopy(self.Global_Model) for i in range(self.numOfAgents)  ] 
        
    # Making same optimizers for all models with similar learning rate
    if learning_rate == 'same':
        self.optimizers = [optim.SGD(cm.parameters(), lr=lr) for cm in self.clientModels]
    
    # Making same optimizers for all models with different learning rate
    elif learning_rate == 'diff':
        choices = np.array([0.05,0.04,0.03,0.02,0.01])
        self.selectedLR = np.random.choice(choices, size = (self.numOfAgents,))
        self.optimizers = [optim.SGD(cm.parameters(), lr=self.selectedLR[i], momentum=0.9) for i,cm in enumerate(self.clientModels)]
    
    self.paramNum()
    self.init_vars()

  def paramNum(self):
    self.paramNumber = count_parameters(self.Global_Model)
    print("Number of parameters:" , self.paramNumber )
    
  def init_vars(self):
    self.AvgGlobalLoss = []
    self.accs = []
    self.Weights = torch.zeros((self.numOfAgents,self.paramNumber), device = self.device)
    self.Losses = [ [] for i in range(self.numOfAgents) ]
    self.mem_wei = [[] for i in range(self.numOfAgents)]
    self.local_accs = [[] for i in range(self.numOfAgents)]
    self.global_acc_avg = [[], []]  
    self.chexpert_acc = [[] for i in range(13)]
    self.avg_acc_chexpert = [[] for i in range(13)]
  def Locals_Accs(self,clientModels, test_data):
    if self.data == 'chexpert':
        return
    # this happens in the testing dataset
    for i,cm in enumerate(clientModels):
        correct = 0
        numoftest = 0
        for adata in test_data:
            P = cm(adata[0][0].to(self.device))
            _, predicted = torch.max(P.data, 1)
            correct += (predicted == adata[0][1].to(self.device)).sum().item()
            numoftest = numoftest + float(adata[0][1].shape[0])
        localacc = (correct/ numoftest)*100
        self.local_accs[i].append(localacc)  
  
  def global_acc(self,testData,rounds):
    # total and correcto to calculate accuracy
    total = 0 
    correct = 0
    z = 0
    count = 0
    if self.data == 'chexpert':
        sig = nn.Sigmoid()
    if self.data == 'noniid_skincancer':
        sig = nn.Softmax(dim=1)
    count_labels = 0
    whole_labels = []
    whole_pred = []
    printer_roc = []
    with torch.no_grad():
        for i, data in enumerate(testData):
            if self.data == 'chexpert':
                images, labels = data['img'].to(self.device), torch.nan_to_num(data['lab']).to(self.device)
            if self.data == 'noniid_skincancer':
                images, labels = data[0]
            images = images.to(self.device)
            labels = labels.to(self.device)
            count_labels = count_labels + 1
            pred = sig(self.Global_Model(images)) 
            whole_labels.append(labels)
            whole_pred.append(pred)
#             _, predicted = torch.max(pred.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
            z = self.criterion(pred,labels) + z
            count+=1
    whole_labels = torch.cat(whole_labels)
    whole_pred = torch.cat(whole_pred)
    
    
    
    if self.data == 'chexpert':
        for i in range(self.numOfClasses):
            roc_val = sklearn.metrics.roc_auc_score(whole_labels[:,i].to('cpu'), whole_pred[:,i].to('cpu'))
            self.rocs[i].append(roc_val)
            printer_roc.append(roc_val)

        print(printer_roc)
        # emptying the array
        self.chexpert_acc = [[] for i in range(13)]
        self.AvgGlobalLoss.append(z.item()/count)
    elif self.data == 'noniid_skincancer': 
        roc_val = sklearn.metrics.roc_auc_score(whole_labels.to('cpu'), whole_pred.to('cpu'), multi_class = 'ovo', average= 'macro')
        print("roc: ",roc_val)
        self.rocs.append(roc_val)
        self.AvgGlobalLoss.append(z.item()/count)
    
#     acc = (100 * correct / total)
#     self.global_acc_avg[0].append(acc)
    if self.print_accuracy == True:
        print(acc,"%")
        print("round", rounds)  
    if len(self.global_acc_avg[0]) == 5:
        self.global_acc_avg[1].append(np.sum(self.global_acc_avg[0])/ 5) 
        self.global_acc_avg[0] = []    
#     self.accs.append(acc)

  def send_to_clients(self):
    with torch.no_grad():
        for cm in self.clientModels:
            for w, cmw in zip(self.Global_Model.parameters(), cm.parameters()):
                cmw.set_(  w.data+0.0 )  
                

  def learn(self,cm, opt, adata):
    random_indices = torch.randperm(self.batch_size)
    x = 0
    if self.data == 'chexpert':
        for i in adata:
            inputs, labels = i['img'].to(self.device), torch.nan_to_num(i['lab']).to(self.device)  
            opt.zero_grad()
            outputs = cm(inputs)
            self.loss = self.criterion(outputs, labels)
            self.loss.backward()
            opt.step()

    else:
        inputs, labels = adata[0][0][random_indices].to(self.device), adata[0][1][random_indices].to(self.device)
    
    if self.data == 'chexpert':
        return
    opt.zero_grad()
    outputs = cm(inputs)
    self.loss = self.criterion(outputs, labels)
    self.loss.backward()
    opt.step()  
    
  def learn_label_flip(self,cm, opt, adata,i):
    random_indices = torch.randperm(self.batch_size)
    if self.data == 'chexpert':
        for i in adata:
            inputs, labels = i['img'].to(self.device), torch.nan_to_num(i['lab']).to(self.device) 
            labs = copy.deepcopy(labels)
            labels[0,:][labs[0,:] == 0] = 1 
            labels[1,:][labs[1,:] == 1] = 0 
            labels[2,:][labs[2,:] == 0] = 1 
            labels[3,:][labs[3,:] == 1] = 0
            opt.zero_grad()
            outputs = cm(inputs)
            self.loss = self.criterion(outputs, labels)
            self.loss.backward()
            opt.step()
    
    else:
        inputs, labels = adata[0][0][random_indices].to(self.device), adata[0][1][random_indices].to(self.device)

    if self.data == 'chexpert':
        return
    opt.zero_grad()
    outputs = cm(inputs)
    self.loss = self.criterion(outputs, labels)
    self.loss.backward()
    opt.step() 

  def Attack(self,agent,rounds):
    if self.attack == False:
        return
    if agent in self.attacked_agents:
        type_of_attack = self.attacked_agents[agent]
        if type_of_attack == 'random_weight':
            self.Weights[agent,:] = torch.rand(self.paramNumber)
        if type_of_attack == 'opposite_weight':
            self.Weights[agent,:] = self.opposites[agent] * self.Weights[agent,:]
        if type_of_attack == 'scaled_weight':
            self.Weights[agent,:] = self.scales[agent] * self.Weights[agent,:]
        if type_of_attack == 'craftedModel' and rounds > 0:
            lambdaz = 0.1
            S = (weight_vec(self.Global_Model) > weight_vec(self.prev_Global_model)).long()
            S[S==0] = -1
            self.Weights[agent,:] = weight_vec(self.Global_Model) - (lambdaz*S)

  def updateW(self,modelDiff,clientMod):
    # Now add the weight for agent 
    if modelDiff == True: 
        self.weight_vector = (weight_vec(clientMod) - weight_vec(self.Global_Model)) 
    else: 
        self.weight_vector = weight_vec(clientMod)  # getting the weight as a vector of that model 
        # Update the Weights matrix  
  

  def abnormalCompute(self):
    Weights_cpu = self.Weights.to('cpu').numpy()
    self.distance_M =  torch.tensor(distance.cdist(Weights_cpu, Weights_cpu, metric='cosine'))
    self.distance_EU = torch.tensor(distance_matrix(Weights_cpu, Weights_cpu))
    self.clf_cs.fit(self.distance_M)
    self.clf_eu.fit(self.distance_EU)
    self.abnormalScores = (self.clf_cs.decision_function(self.distance_M) + self.clf_eu.decision_function(self.distance_EU))
    self.weighted_abnormals = np.exp(self.alpha*self.abnormalScores) / np.sum(np.exp(self.alpha*self.abnormalScores))
    
  def convert_to_tensor(self):
    self.weighted_abnormals = torch.tensor(self.weighted_abnormals, device = self.device)
    self.weighted_abnormals = self.weighted_abnormals.reshape(self.weighted_abnormals.shape[0],1)
    self.Weights = self.weighted_abnormals * self.Weights
    self.G_weight = torch.sum(self.Weights,axis = 0)
    self.G_weight = self.G_weight.type(torch.cuda.FloatTensor)
    
  def COPOD_O(self):
    for rounds in tqdm(range(self.numOfRounds)): 
        for i, (cm, opt, adata, agentLoss) in enumerate(zip(self.clientModels, self.optimizers, self.train_data, self.Losses)): 
            for epoch in range(self.local_steps): 
                # getting the data for that agent
                if i not in self.label_flips:
                    self.learn(cm,opt,adata) # learning
                
                if i in self.label_flips:
                    self.learn_label_flip(cm,opt,adata,i)
                    
            agentLoss.append(self.loss.item()) # appending the loss for each client
            
            # Updating whether with weight or with model difference based on self.modelDiff
            self.updateW(self.modelDiff, cm)            
            self.Weights[i,:] = self.weight_vector
            self.Attack(i,rounds)
        # total_difference = any_anomaly_detection(sel.weigths)  
        if self.scaledWeights == True: self.Weights = self.Weights * self.number
        self.abnormalCompute()
        if self.scaledWeights == True: self.Weights = self.Weights / self.number
        A = 0
        for i,wei in enumerate(self.weighted_abnormals):
            self.mem_wei[i].append(wei+A)
            A = A + 1
        
        if self.print_weights == True:
            print(self.weighted_abnormals)
        
        self.convert_to_tensor() # converting the G_weight to tensor
        # local_models accuracy appending
        self.Locals_Accs(self.clientModels, self.test_data)    
        # Now we need to reshape G_weight back and plug the weights to the global model
        if self.modelDiff == True: 
            AA = self.G_weight + weight_vec(self.Global_Model)
            self.G_weight = AA
        self.prev_Global_model = copy.deepcopy(self.Global_Model)
        self.Global_Model = weight_dec_global(self.Global_Model.to(self.device), self.G_weight, 1) 
        # getting global_model accuracy either in training data or testing 
        self.global_acc(self.test_data, rounds)
        # sending back to the clients
        self.send_to_clients()
        if rounds % 5 == 0:
            self.save_global(rounds)

  def show_plots(self):
    plotsShow(self.numOfAgents,self.AvgGlobalLoss,self.accs,self.mem_wei,self.global_acc_avg, self.dirr)

  def plotLosses(self):
    LossesPlot(self.Losses,self.numOfAgents, self.dirr)
    
  def plotLocalAccs(self):
    LocalAccsPlot(self.numOfAgents, self.local_accs, self.dirr)

  def attack_setting(self,numofAttacked,AttackInfo):
    self.attacked_agents = {}
    self.label_flips = {}
    self.scales = {}
    self.opposites = {}
    for i in AttackInfo: 
        attacked_agent = i
        print(attacked_agent)
        type_of_attack = AttackInfo[i]
        type_of_attackStr = ''.join([i for i in type_of_attack if not i.isdigit()])
        type_of_attackNum = ''.join([i for i in type_of_attack if i.isdigit()])
        if type_of_attackStr == 'scaled_weight': 
            self.scales[attacked_agent] = float(type_of_attackNum)
            
        if type_of_attackStr == 'label_flip':
            original = int(type_of_attackNum[0])
            fake = int(type_of_attackNum[1])
            self.label_flips[attacked_agent] = (original,fake)
            
        if type_of_attackStr == 'opposite_weight':
            self.opposites[attacked_agent] = -1*float(type_of_attackNum)
        
        if type_of_attackStr == 'craftedModel':
            self.crafted == True
        self.attacked_agents[attacked_agent] = type_of_attack

  def save_global(self,rounds):
    path = self.dirr
    if os.path.isdir(path) == False:
        os.mkdir(path)  
    parent_dir = self.dirr
    #self.dirName = str(input("Enter the name of the directory:(MethodName_Rounds_LocalSteps_Attack)"))
    #modelName = str(input("Enter the name of the model followed by .pt e.g model.pt"))
    path = os.path.join(parent_dir, 'model'+ str(rounds) +'.pt')
    torch.save(self.Global_Model.state_dict(), path)
    
  def saveAsPickle(self):
    parent_dir = '/home/naif.alkhunaizi/Desktop/Research/EXP1/Plots/real_non_iid/'+self.dirr
    #self.dirName = str(input("Enter the name of the directory:(MethodName_Rounds_LocalSteps_Attack)"))
    path = os.path.join(parent_dir)
    #os.mkdir(path)
    
    with open(os.path.join(path,'global_acc.pkl'),'wb') as f:
        pkl.dump(self.accs, f)
    
    with open(os.path.join(path,'global_avg_accs.pkl'),'wb') as f:
        pkl.dump(self.global_acc_avg[1], f)
    
    with open(os.path.join(path,'weights.pkl'),'wb') as f:
        pkl.dump(self.mem_wei, f)
    
    with open(os.path.join(path,'avgGlobalLoss.pkl'),'wb') as f:
        pkl.dump(self.AvgGlobalLoss, f)
    
    with open(os.path.join(path,'localLoss.pkl'),'wb') as f:
        pkl.dump(self.Losses, f)

    with open(os.path.join(path,'localAccs.pkl'),'wb') as f:
        pkl.dump(self.local_accs, f)
  
  def chexPickle(self):
    path = self.dirr
    if os.path.isdir(path) == False:
        os.mkdir(path)  
    with open(os.path.join(path,'rocs.pkl'),'wb') as f:
        pkl.dump(self.rocs, f) 
    with open(os.path.join(path,'AvgGlobalLoss.pkl'),'wb') as f:
        pkl.dump(self.AvgGlobalLoss, f) 
    
    if self.data == 'COPOD':
        with open(os.path.join(path,'mem_wei.pkl'),'wb') as f:
            pkl.dump(self.mem_wei, f)  
        
  def Train(self,batch_size,numOfAttacked, AttackInfo,method, attack, local_steps, numOfRounds, print_accuracy = True, print_weights = True, scaleWeights = False):
    self.scaledWeights = scaleWeights
    self.print_accuracy = print_accuracy
    self.local_steps = local_steps 
    self.numOfRounds = numOfRounds
    self.batch_size = batch_size
    self.attack = attack
    self.method = method
    self.AttackInfo = AttackInfo
    self.dirr = self.data + str(self.method) + '_' + str(self.numOfRounds) + '_' + str(self.numOfAgents) + '_' + 'Attack' + str(self.attack) + '_'  + 'MD' + '_' + str(self.modelDiff) + '_seed_'+ str(self.seedo) + str(self.AttackInfo)
    if self.scaledWeights == True:
        self.number = float(input("How much you want to scale all weights?"))
    if self.attack == True and self.Done == False:
        self.crafted = False
        self.attack_setting(numOfAttacked, AttackInfo)
        self.Done = True
    if self.attack == False:
        self.attacked_agents = {}
        self.label_flips = {}
        self.scales = {}
        self.opposites = {} 
    if method == 'COPOD':
#         self.alpha = float(input("Choose value of alpha:"))
        self.alpha = -1
        self.print_weights = print_weights
        self.clf_cs = COPOD()
        self.clf_eu = COPOD()
        self.COPOD_O()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--numOfAgents', type = int, required = True)
    parser.add_argument('--numOfClasses', type = int, required = True)
    parser.add_argument('--data', type = str, required = True, help = 'noniid_skincancer, chexpert')
    parser.add_argument('--modelDiff', type = bool, default = False)
    parser.add_argument('--modelName', type = str, required = True)
    parser.add_argument('--Attack', type = bool, default = False)
    parser.add_argument('--numOfAttacked', type = int, required = True)
    parser.add_argument('--local_steps', type = int, required = True)
    parser.add_argument('--numOfRounds', type = int, required = True)
    parser.add_argument('--AttackInfo', type = str, help = '{index of agent:Attack name |label_flip, random_weight, opposite_weight, scaled_weight,craftedModel ')
    parser.add_argument('--seed', type = int, help = 'to run the experiment with different seeds', required = True)
    parser.add_argument('--lr', type = float, help = 'learning rate for agents', required = True)
    parser.add_argument('--B', type = int, help = 'for batch size for chexpert')
    args = parser.parse_args()
    if args.Attack == True:
        AttInfo = eval(args.AttackInfo)
    if args.Attack == False: 
        AttInfo = None
   

    Fed = Federated(B = args.B, numOfAgents = args.numOfAgents, numOfClasses = args.numOfClasses, data = args.data, modelDiff= args.modelDiff, seedo  = args.seed) # check modDiff if you want it
    Fed.initilize_models(modelName=args.modelName, learning_rate= 'same', lr = args.lr)
    Fed.Train(batch_size = args.B, method = args.method, attack = args.Attack, local_steps = args.local_steps, numOfRounds = args.numOfRounds, print_accuracy = False, scaleWeights = False, numOfAttacked = args.numOfAttacked, AttackInfo = AttInfo)
    Fed.chexPickle()
    Fed.save_global('final')