import os
import sys
import csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import torch

from torch.utils.tensorboard import SummaryWriter

from Dataloader import Dataloader
from Loss_CNN import Loss_CNN
from Classifier import Classifier
from MMD_loss import MMD_loss
from CNN import CNN




def main():

    #unpack arguments for training
    train_params = sys.argv
    num_epochs = train_params[1]
    GAMMA = train_params[2]
    num_pool = train_params[3]
    print(f"Num of epochs: {num_epochs} GAMMA: {GAMMA} Number of pooling layers: {num_pool}")



    #init writer for tensorboard    
    writer_graph = SummaryWriter('runs/Dataloader2/graph')
    writer_source_val = SummaryWriter('runs/Dataloader2/source_val')
    writer_source_mmd = SummaryWriter('runs/Dataloader2/source_mmd')
    writer_source_ce = SummaryWriter('runs/Dataloader2/source_ce')
    writer_target_val = SummaryWriter('runs/Dataloader2/target_val')
    writer_target_mmd = SummaryWriter('runs/Dataloader2/target_mmd')
    writer_target_ce = SummaryWriter('runs/Dataloader2/target_ce')

    writer_source = {}
    writer_source["val"] = writer_source_val
    writer_source["mmd"] = writer_source_mmd
    writer_source["ce"] = writer_source_ce

    writer_target = {}
    writer_target["val"] = writer_target_val
    writer_target["mmd"] = writer_target_mmd
    writer_target["ce"] = writer_target_ce


    #define training params
    #num_epochs = 30
    #GAMMA = 1,8
    SIGMA = torch.tensor([1,2,4,8,16],dtype=torch.float64)

    #mmd_loss_flag
    MMD_loss_flag_phase = {}
    MMD_loss_flag_phase["val"] = False
    MMD_loss_flag_phase["mmd"] = True
    MMD_loss_flag_phase["ce"] = False

    #Models
    input_size = 3
    input_fc_size = 32*299
    hidden_fc_size_1 = 50
    hidden_fc_size_2 = 3
    output_size = 2
    model_cnn =  CNN(input_size, input_fc_size, hidden_fc_size_1)
    model_fc = Classifier(hidden_fc_size_1, hidden_fc_size_2, output_size)


    #define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    MMD_loss_calculator = MMD_loss(fix_sigma = SIGMA)
    loss_cnn = Loss_CNN(model_cnn, model_fc, criterion, MMD_loss_calculator, MMD_loss_flag_phase, GAMMA)


    #Optimizer
    optimizer1 = torch.optim.Adam([
    {'params': model_cnn.parameters()},
    {'params': model_fc.parameters(), 'lr': 1e-2}
    ], lr=1e-2, betas=(0.9, 0.999))

    optimizer2 = torch.optim.Adam(model_fc.parameters(), lr=1e-2, betas=(0.9, 0.999))

    #training iterations
    phases = ['val', 'mmd', 'ce']

    #init variables which collect loss, accuracies for each epoch and train phase
    loss_collected = 0
    source_ce_loss_collected = 0
    target_ce_loss_collected = 0
    mmd_loss_collected = 0
    acc_total_source_collected = 0
    acc_total_target_collected = 0

    #plot lists
    mmd_loss_list = {}
    mmd_loss_list['val']=[]
    mmd_loss_list['mmd']=[]
    mmd_loss_list['ce'] = []

    ce_loss_list_source = {}
    ce_loss_list_source['val']=[]
    ce_loss_list_source['mmd']=[]
    ce_loss_list_source['ce'] = []

    ce_loss_list_target = {}
    ce_loss_list_target['val']=[]
    ce_loss_list_target['mmd']=[]
    ce_loss_list_target['ce'] = []

    accuracy_list_source = {}
    accuracy_list_source['val']=[]
    accuracy_list_source['mmd']=[]
    accuracy_list_source['ce'] = []

    accuracy_list_target = {}
    accuracy_list_target['val']=[]
    accuracy_list_target['mmd']=[]
    accuracy_list_target['ce'] = []

    #Load data
    window_size = 1024
    overlap_size = 0
    features_of_interest = ['C:x_bottom', 'C:y_bottom', 'C:z_bottom']
    list_of_source_BSD_states = ["2", "3", "11", "12", "20", "21"]
    list_of_target_BSD_states = ["5", "6", "14", "15", "23", "24"]
    data_path = Path(os.getcwd()).parents[1]
    data_path = os.path.join(data_path, "data")
    dataloader_split_ce = 0.6
    dataloader_split_mmd = 0.2
    dataloader_split_val = 0.2
    batch_size = 32
    dataloader_source = Dataloader(data_path, list_of_source_BSD_states, window_size, overlap_size, features_of_interest, dataloader_split_ce, dataloader_split_mmd, dataloader_split_val, batch_size)
    dataloader_target = Dataloader(data_path, list_of_target_BSD_states, window_size, overlap_size, features_of_interest, dataloader_split_ce, dataloader_split_mmd, dataloader_split_val, batch_size)
    source_loader = dataloader_source.create_dataloader()
    target_loader = dataloader_target.create_dataloader()

    # Train and Validate the model
    for epoch in range(num_epochs):

        #init array which collects the data in FC for plottnig the data distribution
        class_0_source_fc2_collect = torch.empty((0,3))
        class_1_source_fc2_collect = torch.empty((0,3))
        class_0_target_fc2_collect = torch.empty((0,3))
        class_1_target_fc2_collect = torch.empty((0,3))

        
        for phase in phases:

            #init the dataloader for source and target data for each epoch
            iter_loader_source = iter(source_loader[phase])
            iter_loader_target = iter(target_loader[phase])

            #iterate through phases
            for i in range(len(iter_loader_source)):
                
                ########Forward pass########
                batch_data_source, labels_source = iter_loader_source.next() #batch_size number of windows and labels from source domain
                batch_data_target, labels_target = iter_loader_target.next() #batch_size number of windows from target domain
                batch_data = torch.cat((batch_data_source, batch_data_target), dim=0) #concat the windows to 2*batch_size number of windows



                #validation
                if phase == "val":
                    model_cnn.train(False)
                    model_fc.train(False)
                    
                    with torch.no_grad():
                        _, mmd_loss, source_ce_loss, target_ce_loss, acc_total_source, acc_total_target, class_0_source_fc2, class_1_source_fc2, class_0_target_fc2, class_1_target_fc2 = loss_cnn.forward(batch_data, labels_source, labels_target)
                        
                        # collect latent features for plot 
                        class_0_source_fc2_collect = torch.cat((class_0_source_fc2_collect, class_0_source_fc2), 0)
                        class_1_source_fc2_collect = torch.cat((class_1_source_fc2_collect, class_1_source_fc2), 0)
                        class_0_target_fc2_collect = torch.cat((class_0_target_fc2_collect, class_0_target_fc2), 0)
                        class_1_target_fc2_collect = torch.cat((class_1_target_fc2_collect, class_1_target_fc2), 0)

                #training
                else:
                    model_cnn.train(True)
                    model_fc.train(True)
                    
                    ######## Forward pass ########
                    loss, mmd_loss, source_ce_loss, target_ce_loss, acc_total_source, acc_total_target, _, _, _, _ = loss_cnn.forward(batch_data, labels_source, labels_target)
                    
                    mmd_loss = mmd_loss.detach()
                    source_ce_loss = source_ce_loss.detach()
                    target_ce_loss = target_ce_loss.detach()

                    ######## Backward pass ########
                    if phase == "mmd":
                        optimizer1.zero_grad()
                        loss.backward()
                        optimizer1.step()
                    elif phase == "ce":
                        optimizer2.zero_grad()
                        loss.backward()
                        optimizer2.step()

                
                #collect loss, accuracies over an epoch
                mmd_loss_collected += mmd_loss
                source_ce_loss_collected += source_ce_loss
                target_ce_loss_collected += target_ce_loss
                acc_total_source_collected += acc_total_source
                acc_total_target_collected += acc_total_target
                
                
            
            # data distribution
            if phase == "val" and (epoch ==0 or epoch ==5 or epoch == 10 or epoch ==20):
                
                #define ffigure
                fig = plt.figure()
                plt.gcf().set_size_inches((20, 20)) 
                ax = fig.add_subplot(projection='3d')


                #plot data
                m = [1,2,3,4]
                data = [class_0_source_fc2_collect, class_1_source_fc2_collect, class_0_target_fc2_collect, class_1_target_fc2_collect]
                for i in range(4):
                    ax.scatter(data[i][:,0], data[i][:,1], data[i][:,2], marker=m[i])
                
                #safe data
                b = open(f'{epoch}_distribution.csv', 'w')
                a = csv.writer(b)
                a.writerows(data)
                b.close()
                
                #label axis
                ax.set_xlabel('Neuron 1 $\longrightarrow$', rotation=0, labelpad=10, size=20)
                ax.set_ylabel('Neuron 2 $\longrightarrow$', rotation=0, labelpad=10, size=20)
                ax.set_zlabel('Neuron 3 $\longrightarrow$', rotation=0, labelpad=10, size=20)
                plt.rcParams.update({'font.size': 10})
                
                #show and safe fig
                fig.savefig(f"data_distribution_{epoch}", format='pdf')              

            
            # Normalize collected loss, accuracies for each epoch and train phase
            running_mmd_loss = mmd_loss_collected / len(source_loader[phase])
            running_acc_source = acc_total_source_collected / len(source_loader[phase])
            running_acc_target = acc_total_target_collected / len(target_loader[phase])
            running_source_ce_loss = source_ce_loss_collected / len(source_loader[phase])
            running_target_ce_loss = target_ce_loss_collected / len(target_loader[phase])
            
            
            #Add train data to plot list
            accuracy_list_source[phase].append(running_acc_source)
            accuracy_list_target[phase].append(running_acc_target)
            ce_loss_list_source[phase].append(running_source_ce_loss)
            ce_loss_list_target[phase].append(running_target_ce_loss)
            mmd_loss_list[phase].append(running_mmd_loss)


            #Add train data to tensorboard list
            writer_source[phase].add_scalar(f'accuracy', running_acc_source, epoch)
            writer_target[phase].add_scalar(f'accuracy', running_acc_target, epoch)
            writer_source[phase].add_scalar(f'ce_loss', running_source_ce_loss, epoch)
            writer_target[phase].add_scalar(f'ce_loss', running_target_ce_loss, epoch)
            writer_source[phase].add_scalar(f'mmd_loss', running_mmd_loss, epoch)
            

            #Reset variable for collected loss, accuracies for each epoch and train phase
            loss_collected = 0
            source_ce_loss_collected = 0
            target_ce_loss_collected = 0
            mmd_loss_collected = 0
            acc_total_source_collected = 0
            acc_total_target_collected = 0
                

        print(f"Epoch {epoch+1}/{num_epochs} successfull")

    #Plot training curves
    fig1 = plt.figure()
    plt.title('Accuracy Source Domain')
    plt.plot(accuracy_list_source['ce'], 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
    plt.plot(accuracy_list_source['mmd'], 'ro-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
    plt.plot(accuracy_list_source['val'], 'go-', label = 'Val', linewidth=1,markersize=0.1)
    plt.xlabel("Epoch $\longrightarrow$")
    plt.ylabel("Accuracy Source Domain $\longrightarrow$")
    plt.legend()
    fig1.savefig('Accuracy Source Domain', format='pdf')
    pd.DataFrame(accuracy_list_source['ce']).to_csv('accuracy_list_source_ce.csv',index=False,header=False)
    pd.DataFrame(accuracy_list_source['mmd']).to_csv('accuracy_list_source_mmd.csv',index=False,header=False)
    pd.DataFrame(accuracy_list_source['val']).to_csv('accuracy_list_source_val.csv',index=False,header=False)

    fig2 = plt.figure()
    plt.title('Accuracy Target Domain')
    plt.plot(accuracy_list_target['ce'], 'co-', label = 'CE-Loss', linewidth=1,markersize=0.1)
    plt.plot(accuracy_list_target['mmd'], 'mo-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
    plt.plot(accuracy_list_target['val'], 'yo-', label = 'Val', linewidth=1,markersize=0.1)
    plt.xlabel("Epoch $\longrightarrow$")
    plt.ylabel("Accuracy Target Domain $\longrightarrow$")
    plt.legend()
    fig2.savefig('Accuracy Target Domain', format='pdf')
    pd.DataFrame(accuracy_list_target['ce']).to_csv('accuracy_list_target_ce.csv',index=False,header=False)
    pd.DataFrame(accuracy_list_target['mmd']).to_csv('accuracy_list_target_mmd.csv',index=False,header=False)
    pd.DataFrame(accuracy_list_target['val']).to_csv('accuracy_list_target_val.csv',index=False,header=False)

    fig3 = plt.figure()
    plt.title('CE-Loss Source Domain')
    plt.plot(ce_loss_list_source['ce'], 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
    plt.plot(ce_loss_list_source['mmd'], 'ro-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
    plt.plot(ce_loss_list_source['val'], 'go-', label = 'Val', linewidth=1,markersize=0.1)
    plt.xlabel("Epoch $\longrightarrow$")
    plt.ylabel("CE-Loss Source Domain $\longrightarrow$")
    plt.legend()
    fig3.savefig('CE_Loss Source Domain', format='pdf')
    pd.DataFrame(ce_loss_list_source['ce']).to_csv('ce_loss_list_source_ce.csv',index=False,header=False)
    pd.DataFrame(ce_loss_list_source['mmd']).to_csv('ce_loss_list_source_mmd.csv',index=False,header=False)
    pd.DataFrame(ce_loss_list_source['val']).to_csv('ce_loss_list_source_val.csv',index=False,header=False)

    fig4 = plt.figure()
    plt.title('CE-Loss Target Domain')
    plt.plot(ce_loss_list_target['ce'], 'co-', label = 'CE-Loss', linewidth=1,markersize=0.1)
    plt.plot(ce_loss_list_target['mmd'], 'mo-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
    plt.plot(ce_loss_list_target['val'], 'yo-', label = 'Val', linewidth=1,markersize=0.1)
    plt.xlabel("Epoch $\longrightarrow$")
    plt.ylabel("CE-Loss Target Domain $\longrightarrow$")
    plt.legend()
    fig4.savefig('CE_Loss Target Domain', format='pdf')
    pd.DataFrame(ce_loss_list_target['ce']).to_csv('ce_loss_list_target_ce.csv',index=False,header=False)
    pd.DataFrame(ce_loss_list_target['mmd']).to_csv('ce_loss_list_target_mmd.csv',index=False,header=False)
    pd.DataFrame(ce_loss_list_target['val']).to_csv('ce_loss_list_target_val.csv',index=False,header=False)

    fig5 = plt.figure()
    plt.title('MMD-Loss')
    plt.plot(mmd_loss_list['ce'], 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
    plt.plot(mmd_loss_list['mmd'], 'ro-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
    plt.plot(mmd_loss_list['val'], 'go-', label = 'Val', linewidth=1,markersize=0.1)
    plt.xlabel("Epoch $\longrightarrow$")
    plt.ylabel("MMD-Loss $\longrightarrow$")
    plt.legend()
    fig5.savefig('MMD_Loss', format='pdf')
    pd.DataFrame(mmd_loss_list['ce']).to_csv('mmd_loss_list_ce.csv',index=False,header=False)
    pd.DataFrame(mmd_loss_list['mmd']).to_csv('mmd_loss_list_mmd.csv',index=False,header=False)
    pd.DataFrame(mmd_loss_list['val']).to_csv('mmd_loss_list_val.csv',index=False,header=False)

if __name__ == "__main__":
    main()