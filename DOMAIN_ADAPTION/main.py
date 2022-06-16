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
from Plotter import Plotter

def main():

    #unpack arguments for training
    train_params = sys.argv[1:]
    features_of_interest = train_params[0]
    num_epochs = int(train_params[1])
    GAMMA = float(train_params[2])
    num_pool = int(train_params[3])
    print(f"Features of interest: {features_of_interest} Num of epochs: {num_epochs} GAMMA: {GAMMA} num_pool: {num_pool}" )

    #Folder name to store data for each experiment
    features_of_interest_folder = features_of_interest.replace("/", "_")
    folder_to_store_data = "feature=" + str(features_of_interest_folder) + "_" + "num_epochs=" + str(num_epochs) + "_" + "GAMMA=" + str(GAMMA)

    #Generate folder structure to store plots and data
    current_directory = os.getcwd()
    path_learning_curve = os.path.join(current_directory, folder_to_store_data, "learning_curve")
    path_learning_curve_data = os.path.join(current_directory, folder_to_store_data, "learning_curve_data")
    path_data_distribution = os.path.join(current_directory, folder_to_store_data, "data_distribution")
    path_data_distribution_data = os.path.join(current_directory, folder_to_store_data, "data_distribution_data")
    path_accuracy = os.path.join(current_directory, folder_to_store_data, "accuracy")

    if not os.path.exists(path_learning_curve): #Folder to store Learning Curve Plots 
        os.makedirs(path_learning_curve)
    if not os.path.exists(path_learning_curve_data): #Folder to store Learning Curve Plots Data
        os.makedirs(path_learning_curve_data)
    if not os.path.exists(path_data_distribution): #Folder to store Data Distribuiton Plots 
        os.makedirs(path_data_distribution)
    if not os.path.exists(path_data_distribution_data): #Folder to store Data Distribuiton Plots Data 
        os.makedirs(path_data_distribution_data)
    if not os.path.exists(path_accuracy): #Folder to store Accuracies of Training
        os.makedirs(path_accuracy)

    #init plotter for generating plots from data
    plotter = Plotter(folder_to_store_data)

    # create csv file to store data 
    f_learning_curve = open(f'{folder_to_store_data}/learning_curve_data/learning_curve.csv', 'w')
    f_accuracy = open(f'{folder_to_store_data}/accuracy/accuracies.csv', 'w')

    # create csv writer to store data
    f_learning_curve_writer = csv.writer(f_learning_curve)
    f_learning_curve_writer.writerow(['running_acc_source_val','running_acc_target_val','running_source_ce_loss_val','running_target_ce_loss_val','running_mmd_loss_val','running_acc_source_mmd','running_acc_target_mmd','running_source_ce_loss_mmd','running_target_ce_loss_mmd','running_mmd_loss_mmd','running_acc_source_ce','running_acc_target_ce','running_source_ce_loss_ce','running_target_ce_loss_ce','running_mmd_loss_ce'])

    #header for csv file
    f_accuracy_writer = csv.writer(f_accuracy)
    f_accuracy_writer.writerow(['accuracy_source_val','accuracy_target_val','accuracy_source_mmd','accuracy_target_mmd','accuracy_source_ce','accuracy_target_ce'])

    #training iterations
    phases = ['val', 'mmd', 'ce']

    #init writer for tensorboard    
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

    #Windowing details
    window_size = 1024
    overlap_size = 0

    # Define which BSD states should be included in source and target domain
    list_of_source_BSD_states = ["2", "3"]#, "11", "12", "20", "21"]
    list_of_target_BSD_states = ["5", "6"]#, "14", "15", "23", "24"]

    # Path where dataset is stored
    data_path = Path(os.getcwd()).parents[1]
    data_path = os.path.join(data_path, "data")

    # Dataloader
    dataloader_split_ce = 0.6
    dataloader_split_mmd = 0.2
    dataloader_split_val = 0.2
    batch_size = 32
    dataloader_source = Dataloader(data_path, list_of_source_BSD_states, window_size, overlap_size, features_of_interest, dataloader_split_ce, dataloader_split_mmd, dataloader_split_val, batch_size)
    dataloader_target = Dataloader(data_path, list_of_target_BSD_states, window_size, overlap_size, features_of_interest, dataloader_split_ce, dataloader_split_mmd, dataloader_split_val, batch_size)
    source_loader = dataloader_source.create_dataloader()
    target_loader = dataloader_target.create_dataloader()

    #define Sigma for MMD Loss
    SIGMA = torch.tensor([1,2,4,8,16],dtype=torch.float64)

    #Define mmd_loss_flag to specify which trainingphases use MMD and CE Loss
    MMD_loss_flag_phase = {}
    MMD_loss_flag_phase["val"] = False
    MMD_loss_flag_phase["mmd"] = True
    MMD_loss_flag_phase["ce"] = False

    #Models
    input_size = 1
    hidden_fc_size_1 = 50
    hidden_fc_size_2 = 3
    output_size = 2
    model_cnn =  CNN(input_size, hidden_fc_size_1, num_pool, window_size)
    model_fc = Classifier(hidden_fc_size_1, hidden_fc_size_2, output_size)

    # Define Loss
    criterion = torch.nn.CrossEntropyLoss()
    MMD_loss_calculator = MMD_loss(fix_sigma = SIGMA)
    loss_cnn = Loss_CNN(model_cnn, model_fc, criterion, MMD_loss_calculator, MMD_loss_flag_phase, GAMMA)

    #Optimizer
    optimizer1 = torch.optim.Adam([
    {'params': model_cnn.parameters()},
    {'params': model_fc.parameters(), 'lr': 1e-2}
    ], lr=1e-2, betas=(0.9, 0.999))

    optimizer2 = torch.optim.Adam(model_fc.parameters(), lr=1e-2, betas=(0.9, 0.999))

    # Init variables which collect loss, accuracies for each epoch and train phase
    source_ce_loss_collected = 0
    target_ce_loss_collected = 0
    mmd_loss_collected = 0
    acc_total_source_collected = 0
    acc_total_target_collected = 0

    # Train and Validate the model
    for epoch in range(num_epochs):

        #init array which collects the data in FC for plottnig the data distribution
        class_0_source_fc2_collect = torch.empty((0,3))
        class_1_source_fc2_collect = torch.empty((0,3))
        class_0_target_fc2_collect = torch.empty((0,3))
        class_1_target_fc2_collect = torch.empty((0,3))

        #init list to collect data which is stored in one csv file line
        f_accuracy_collect = []
        learning_curve_data_collect = []

        #iterate through phases
        for phase in phases:

            #init the dataloader for source and target data for each epoch
            iter_loader_source = iter(source_loader[phase])
            iter_loader_target = iter(target_loader[phase])

            #iterate through batches of phase specific dataloader
            for _ in range(len(iter_loader_source)):
                
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
                        
                        # collect latent features of fc2 for plot 
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
            
            # store data distribution in latent feature space fc2 in csv
            if phase == "val" and (epoch ==0 or epoch ==1 or epoch == 10 or epoch ==20):
                
                df1 = pd.DataFrame({'class_0_source_fc2_collect_0_dim':class_0_source_fc2_collect[:, 0]})
                df2 = pd.DataFrame({'class_0_source_fc2_collect_1_dim':class_0_source_fc2_collect[:, 1]})
                df3 = pd.DataFrame({'class_0_source_fc2_collect_2_dim':class_0_source_fc2_collect[:, 2]})
                df4 = pd.DataFrame({'class_1_source_fc2_collect_0_dim':class_1_source_fc2_collect[:, 0]})
                df5 = pd.DataFrame({'class_1_source_fc2_collect_1_dim':class_1_source_fc2_collect[:, 1]})
                df6 = pd.DataFrame({'class_1_source_fc2_collect_2_dim':class_1_source_fc2_collect[:, 2]})
                df7 = pd.DataFrame({'class_0_target_fc2_collect_0_dim':class_0_target_fc2_collect[:, 0]})
                df8 = pd.DataFrame({'class_0_target_fc2_collect_1_dim':class_0_target_fc2_collect[:, 1]})
                df9 = pd.DataFrame({'class_0_target_fc2_collect_2_dim':class_0_target_fc2_collect[:, 2]})
                df10 = pd.DataFrame({'class_1_target_fc2_collect_0_dim':class_1_target_fc2_collect[:, 0]})
                df11 = pd.DataFrame({'class_1_target_fc2_collect_1_dim':class_1_target_fc2_collect[:, 1]})
                df12 = pd.DataFrame({'class_1_target_fc2_collect_2_dim':class_1_target_fc2_collect[:, 2]})
                pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12],axis=1).to_csv(f'{folder_to_store_data}/data_distribution_data/data_distribution_{epoch}.csv', index = False)

            # Normalize collected loss, accuracies for each epoch and train phase
            running_mmd_loss = mmd_loss_collected / len(source_loader[phase])
            running_acc_source = acc_total_source_collected / len(source_loader[phase])
            running_acc_target = acc_total_target_collected / len(target_loader[phase])
            running_source_ce_loss = source_ce_loss_collected / len(source_loader[phase])
            running_target_ce_loss = target_ce_loss_collected / len(target_loader[phase])

            #Add train data to tensorboard list
            writer_source[phase].add_scalar(f'accuracy', running_acc_source, epoch)
            writer_target[phase].add_scalar(f'accuracy', running_acc_target, epoch)
            writer_source[phase].add_scalar(f'ce_loss', running_source_ce_loss, epoch)
            writer_target[phase].add_scalar(f'ce_loss', running_target_ce_loss, epoch)
            writer_source[phase].add_scalar(f'mmd_loss', running_mmd_loss, epoch)

            #Reset variable for collected loss, accuracies for each epoch and train phase
            source_ce_loss_collected = 0
            target_ce_loss_collected = 0
            mmd_loss_collected = 0
            acc_total_source_collected = 0
            acc_total_target_collected = 0

            #collect data which is stored in one line of csv
            learning_curve_data_collect = learning_curve_data_collect + [running_acc_source, running_acc_target, running_source_ce_loss, running_target_ce_loss, running_mmd_loss]
            f_accuracy_collect = f_accuracy_collect + [running_acc_source, running_acc_target]

        #store write one line in csv 
        f_learning_curve_writer.writerow([running_acc_source, running_acc_target, running_source_ce_loss, running_target_ce_loss, running_mmd_loss])
        f_accuracy_writer.writerow(f_accuracy_collect)
        
        print(f"Epoch {epoch+1}/{num_epochs} successfull")

    #close csv writer for accuracy and learning curves
    f_accuracy.close()
    f_learning_curve.close()

    #plot learning curves and data distribtuion from csv files
    plotter.plot_distribution()
    plotter.plot_curves()

if __name__ == "__main__":
    main()