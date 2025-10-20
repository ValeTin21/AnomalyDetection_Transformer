#Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

import torch
import torchvision
import torch.nn as nn
import torchmetrics
from torch import optim

from sklearn.metrics import roc_curve, auc


#########################
#Creating Checkpoints to save model parameters during training
class SaveBestModel:
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss,epoch, model, optimizer, criterion,numLayer,numHead,LR):
        filename='ModelScan/Final/best_modelL'+str(numLayer)+'H'+str(numHead)+str(LR)+'.pth'
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'layer':numLayer,
                'head': numHead,
                'LR' : LR,
                }, filename)
            
######################################            
#Defining a training loop function

def trainingLoop(train_loader,val_loader,model,device,epochs,loss_func,numLayer,numHead,LR,schedulerSet=False):
    hist_loss = []
    hist_vloss = []
    model.to(device)
    loss_func.to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[25,50,75], gamma=0.1)

    save_model = SaveBestModel()
            
    for epoch in range(epochs):
        t0 = time.time()

        model.train()

        train_loss = 0
        counter = 0
        for xb,_ in train_loader:
            counter += 1
            xb=xb.to(device)

            pred = model(xb)
            loss = loss_func(pred, xb)

            train_loss += loss.item()

            # backpropagation and weights update
            loss.backward()
            opt.step()
            opt.zero_grad()

        train_loss /= counter
        hist_loss.append(train_loss)

        # evaluation step
        model.eval()
        vali_loss = 0
        counter = 0

        with torch.no_grad():
            for xb, _ in val_loader:
                counter += 1
                xb=xb.to(device)

                pred = model(xb)

                vloss = loss_func(pred, xb)
                vali_loss += vloss.item()

        vali_loss/=counter
        hist_vloss.append(vali_loss)

        #save best model
        save_model(vali_loss, epoch, model, opt, loss_func,numLayer,numHead,LR)
        
        if schedulerSet:
            current_lr = lr_scheduler.get_last_lr()[0]
            print("epoch: %d, time(s): %.2f, train loss: %.6f, vali loss: %.6f,lr : %1.2e" % (epoch+1, time.time()-t0, train_loss, vali_loss,current_lr))
            lr_scheduler.step()
        else:
            print("epoch: %d, time(s): %.2f, train loss: %.6f, vali loss: %.6f" % (epoch+1, time.time()-t0, train_loss, vali_loss))
        
    return hist_loss,hist_vloss
    
######################################            
def trainingPlot(hist_loss,hist_vloss,numHead,numLayer,LR):
    plt.figure()
    filename='ModelScan/Final/best_modelL'+str(numLayer)+'H'+str(numHead)+str(LR)+'.pth'
    checkpoint = torch.load(filename)
    plt.plot(range(1,len(hist_loss)+1), hist_loss,"g.-", label='train loss')
    plt.plot(range(1,len(hist_vloss)+1), hist_vloss, "b.-", label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss during training, best at epoch: '+str( checkpoint['epoch'])+' with LR '+str(LR))
    plt.legend()
    savename='ModelScan/Final/Plots/TrainingLoopLossL'+str(numLayer)+'H'+str(numHead)+str(LR)+'.pdf'
    plt.savefig(savename)    
    plt.show()
    
#####################################
def testPlot(bckAS,sgnAS,numHead,numLayer,LR):
    plt.figure()
    plt.title("Unsepervised Transformer over test set")
    plt.hist(bckAS, bins=100, label='Background', alpha=0.6,density=True)
    plt.hist(sgnAS, bins=100, label='Signal', alpha=0.6,density=True)
    plt.xlabel("Anomaly score")
    plt.legend()
    savename='ModelScan/Final/Plots/AnomalyScore'+str(numLayer)+'H'+str(numHead)+str(LR)+'.pdf'
    plt.savefig(savename)    
    plt.show()
    
####################
def ROCaucPlot(labelsAS,anomalyScore,numHead,numLayer,LR):
    plt.figure()
    fpr, tpr, threshold = roc_curve(labelsAS,anomalyScore)
    auc1 = auc(fpr, tpr)
    
    if auc1<(1-auc1):
        auc1=(1-auc1)
        temp=tpr
        tpr=fpr
        fpr=temp
        
    plt.plot(fpr,tpr,label='AUC = %.1f%%'%(auc1*100.))
    plt.axline((0,0),slope=1,color="k",linestyle="--",alpha=0.5)
    plt.xlabel("bkg. mistag rate")
    plt.ylabel("sig. efficiency")
    plt.grid(True)
    plt.title("ROC Curve and AUC Score")
    plt.legend(loc='upper left',fontsize=12,fancybox=True, framealpha=0.9, shadow=True, borderpad=0.5,edgecolor='black', frameon=True)
    savename='ModelScan/Final/Plots/AUCL'+str(numLayer)+'H'+str(numHead)+str(LR)+'.pdf'
    plt.savefig(savename)
    plt.show()

######################
#defining Test Loss Func
def testLossFunc(x,xhat):
    return np.linalg.norm(xhat-x, ord=None)

######################################
def testLoop(test_loader,model,device,numHead,numLayer,LR):
    
    # Running the best model on the test set
    counter=0
    labelsAS=np.array([])
    anomalyScore = []

    with torch.no_grad():
        for xb, yb in test_loader:
            counter += 1
            xb=xb.to(device)
            yb=yb.numpy()
            labelsAS=np.append(labelsAS,yb)
            pred = model(xb)
                        
            for j in range (0,(pred.shape)[0]):
                tloss=0
                xhat=pred[j,:,:]
                xtrue=xb[j,:,:]
                xhat=xhat.cpu()
                xtrue=xtrue.cpu()
                
                tloss = testLossFunc(xtrue, xhat)
                anomalyScore.append(tloss.item())

    print('Test loss list dimension: ', len(anomalyScore))
    
    anomalyScore=np.array(anomalyScore)
    print("SumAnomalyScore^2",np.sum(anomalyScore**2))
    
    labelsAS = (np.rint(labelsAS)).astype(int)
    bckAS=anomalyScore[labelsAS==0]
    sgnAS=anomalyScore[labelsAS==1]
    # print(bckAS.shape,sgnAS.shape)  
    
    ROCaucPlot(labelsAS,anomalyScore,numHead,numLayer,LR) 
    
    testPlot(bckAS,sgnAS,numHead,numLayer,LR) 
