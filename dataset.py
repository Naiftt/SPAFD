import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.backends.backend_pdf import PdfPages
import os.path
from os import path
from skincancer_read import get_data, compute_img_mean_std, CustomDataset
import glob
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import torchxrayvision as xrv

    

def LocalAccsPlot(numOfAgents,local_accs, dirr):
    pathh = dirr
    if path.exists(pathh) == False:
        os.mkdir(pathh)
    pp = PdfPages(pathh+'/3.pdf')
    fig, (ax1, ax2) = plt.subplots(1,2);fig.set_size_inches(25, 6)
    agents = []
    for i in range(0,numOfAgents//2): 
        ax1.plot(local_accs[i]);ax1.set_ylabel("Test Accuracy",fontsize=20);ax1.set_xlabel("Rounds",fontsize=20) 
        agents.append('A'+str(i))
    ax1.legend(agents)
    agents = []
    for i in range(numOfAgents//2,numOfAgents):
        ax2.plot(local_accs[i]);ax2.set_ylabel("Test Accuracy",fontsize=20);ax2.set_xlabel("Rounds",fontsize=20) 
        agents.append('A'+str(i))
    ax2.legend(agents)
    pp.savefig(fig)
    pp.close()

    
def plotsShow(numOfAgents,AvgGlobalLoss,accs,mem_wei,global_acc_avg,dirr):
    pathh = dirr
    if path.exists(pathh) == False:
        os.mkdir(pathh)
    pp = PdfPages(pathh+'/1.pdf')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4);fig.set_size_inches(25, 6)
    ax1.plot(AvgGlobalLoss);ax1.set_ylabel("Avg Global Loss",fontsize=20);ax1.set_xlabel("Rounds",fontsize=20)  
    ax2.plot(accs);ax2.set_ylabel("Global Model Test Accuracy",fontsize=20);ax2.set_xlabel("Rounds",fontsize=20) 
    leg = []
    ax3.set_yticks(np.arange(0,numOfAgents))
    for i in range(numOfAgents):
        ax3.plot(mem_wei[i]);ax3.set_ylabel("Weights For Each Agent",fontsize=20);ax3.set_xlabel("Rounds",fontsize=20)
        leg.append('A'+str(i))
    ax3.legend(leg, loc = 'upper center',  bbox_to_anchor=(0.5, 1.2), ncol=3, fancybox=True)
    if len(global_acc_avg) > 2:
        ax4.plot(global_acc_avg);ax4.set_ylabel("Average Global Model Test Accuracy",fontsize=20);ax4.set_xlabel("Rounds",fontsize=20)
        ax4.set_yticks(np.arange(0,100,10))
    else:
        ax4.plot(global_acc_avg[1]);ax4.set_ylabel("Average Global Model Test Accuracy",fontsize=20);ax4.set_xlabel("Rounds",fontsize=20)
        ax4.set_yticks(np.arange(0,100,10))
    pp.savefig(fig)
    pp.close()

    
def LossesPlot(Losses,numOfAgents,dirr):
    pathh = dirr
    if path.exists(pathh) == False:
        os.mkdir(pathh)
    pp = PdfPages(pathh+'/2.pdf')
    fig, (ax1, ax2) = plt.subplots(1,2);fig.set_size_inches(25, 6)
    agents = []
    for i in range(0,numOfAgents//2): 
        ax1.plot(Losses[i]);ax1.set_ylabel("Losses",fontsize=20);ax1.set_xlabel("Rounds",fontsize=20) 
        agents.append('A'+str(i))
    ax1.legend(agents)
    agents = []
    for i in range(numOfAgents//2,numOfAgents):
        ax2.plot(Losses[i]);ax2.set_ylabel("Losses",fontsize=20);ax2.set_xlabel("Rounds",fontsize=20) 
        agents.append('A'+str(i))
    ax2.legend(agents)
    pp.savefig(fig)
    pp.close()
    




def skinCancer(numOfAgents, input_size = 224, base_dir = '/home/naif_alkhunaizi/Desktop/exps/noAttack/final_repo/skincancer/data'):
    # change base_dir to your path 
    all_image_path = glob.glob(os.path.join(base_dir, '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    df_train, df_val = get_data(base_dir, imageid_path_dict)
    #normMean,normStd = compute_img_mean_std(all_image_path)
    normMean = [0.76303697, 0.54564005, 0.57004493]
    normStd = [0.14092775, 0.15261292, 0.16997]
    train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                          transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                          transforms.ToTensor(),
                                          transforms.Normalize(normMean, normStd)])
    # define the transformation of the val images.
    val_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                        transforms.ToTensor(),
                                        transforms.Normalize(normMean, normStd)])
    training_set = CustomDataset(df_train.drop_duplicates('image_id'), transform=train_transform)
    train_loader = DataLoader(training_set, batch_size=8900//numOfAgents, shuffle=True, num_workers=4)
    # Same for the validation set:
    validation_set = CustomDataset(df_val.drop_duplicates('image_id'), transform=train_transform)
    val_loader = DataLoader(validation_set, batch_size=64, shuffle=False, num_workers=8)
    
    return train_loader, val_loader




def drich_non_skin(numOfAgents,data):
    # static way of creating non iid data, to change the distribution change the index of p in
    # the for loop 
    nonIID_tensors = [[] for i in range(numOfAgents)]  
    nonIID_labels = [[] for i in range(numOfAgents)]  
    agents = np.arange(0,numOfAgents)
    c = 0
    p = np.ones((numOfAgents))
    xx = 0
    for i in data:
        xx+=1
        p = np.ones((numOfAgents))
        if float(i[1]) == 0:
            p[0] = numOfAgents
            p[1] = numOfAgents
            p[2] = numOfAgents
        if float(i[1]) == 1:
            p[0] = numOfAgents
            p[1] = numOfAgents
            p[2] = numOfAgents
        if float(i[1]) == 2:
            p[3] = numOfAgents
            p[8] = numOfAgents
            p[2] = numOfAgents
        if float(i[1]) == 3:
            p[0] = numOfAgents
            p[4] = numOfAgents
            p[5] = numOfAgents
        if float(i[1]) == 4:
            p[3] = numOfAgents
            p[4] = numOfAgents
            p[5] = numOfAgents
        if float(i[1]) == 5:
            p[3] = numOfAgents
            p[4] = numOfAgents
            p[5] = numOfAgents
        if float(i[1]) == 6:
            p[6] = numOfAgents
            p[9] = numOfAgents
            p[8] = numOfAgents
        p = p / np.sum(p)
        j = np.random.choice(agents, p = p)
        nonIID_tensors[j].append(i[0])
        nonIID_labels[j].append(torch.tensor(i[1]).reshape(1))
        if len(data) == 8912 and xx == 8910:
            break
        if len(data) < 8912 and xx == 1100:
            break
    
    dataset = [[] for i in range(numOfAgents) ]
    for i in range(numOfAgents):
        dataset[i].append([torch.stack(nonIID_tensors[i]),torch.cat(nonIID_labels[i])])
    
    return dataset

def chexpert(batch_size, imgpath = "/home/naif_alkhunaizi/CheXpert-v1.0-small", csvpath = "/home/naif_alkhunaizi/CheXpert-v1.0-small/train.csv"):
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
    dataset = xrv.datasets.CheX_Dataset(imgpath= imgpath,
                                       csvpath= csvpath,
                                     views=["PA","AP"], unique_patients=False, transform = transform)
    len_dataset = len(dataset)
    Agents = []
    p = round(1/11 * len(dataset))
    s = [p,p,p,p,p,p,p-1,p-1,p-1,p-1,p-1]
    s  = torch.utils.data.random_split(dataset, s)
    for i in range(10):
        Agents.append(torch.utils.data.DataLoader(s[i], batch_size=batch_size,shuffle=True, num_workers=4))
    test_dataset = torch.utils.data.DataLoader(s[10], batch_size=batch_size,shuffle=True, num_workers=4)
    return Agents, test_dataset

DATASETS = [
    "MosMed",
    "kits",
    "LiTs",
    "RSPECT",
    "IHD_Brain",
    "ImageCHD",
    "CTPancreas",
    "Brain_MRI",
    "ProstateMRI",
    "RSNAXRay",
    "Covid19XRay",
    "ultrasound_covid",
    "fetal_ultrasound",
]

MODES = ["Classify", "Segment"]
SLICE_EXT = ".pt"
SLICES_DIR = "slices"

CLASSES = {
    "Segment": {
        "MosMed": {"Pos": [1]},
        "kits": {"Benign": [1], "Malignant": [2]},
        "LiTs": {"Tumor": [1]},
        "ImageCHD": {
            "Heart": [1, 2, 3, 4, 5, 6, 7],
            "CHD": [8, 9, 10, 11, 12, 13, 14, 15],
        },
        "CTPancreas": [],
    },
    "Classify": {
        "MosMed": ["Neg", "Pos"],
        "kits": ["Benign", "Malignant"],
        "LiTs": ["No_Tumor", "Tumor"],
        "RSPECT": ["No_PE", "PE"],
        "IHD_Brain": ["No_IHD", "IHD"],
        "IHD_Brain_Multi": [
            "No_IHD",
            "epidural",
            "intraparenchymal",
            "intraventricular",
            "subarachnoid",
            "subdural",
        ],
        "ImageCHD": ["No_CHD", "CHD"],
        "CTPancreas": ["No_Tumor", "Tumor"],
        "Brain_MRI": [
            "glioma_tumor",
            "meningioma_tumor",
            "no_tumor",
            "pituitary_tumor",
        ],
    },
}


def skinCancer_drich(numOfAgents, input_size = 128, base_dir = '/home/naif_alkhunaizi/Desktop/exps/noAttack/final_repo/skincancer/data'):
    all_image_path = glob.glob(os.path.join(base_dir, '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    df_train, df_val = get_data(base_dir, imageid_path_dict)
    #normMean,normStd = compute_img_mean_std(all_image_path)
    normMean = [0.76303697, 0.54564005, 0.57004493]
    normStd = [0.14092775, 0.15261292, 0.16997]
    train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                          transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                          transforms.ToTensor(),
                                          transforms.Normalize(normMean, normStd)])
    # define the transformation of the val images.
    val_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                        transforms.ToTensor(),
                                        transforms.Normalize(normMean, normStd)])
    training_set = CustomDataset(df_train.drop_duplicates('image_id'), transform=train_transform)
    train_loader = DataLoader(training_set, batch_size=8900//numOfAgents, shuffle=True, num_workers=4)
    # Same for the validation set:
    validation_set = CustomDataset(df_val.drop_duplicates('image_id'), transform=train_transform)
    val_loader = DataLoader(validation_set, batch_size=1200, shuffle=False, num_workers=8)
    
    return training_set, validation_set