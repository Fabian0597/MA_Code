import os
import sys
import csv
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import random
from datetime import datetime
import torch

from torch.utils.tensorboard import SummaryWriter

#variant1
from Dataloader import Dataloader

from Loss_CNN import Loss_CNN
from Classifier import Classifier
from MMD_loss import MMD_loss
from MMD_loss_CNN import MMD_loss_CNN

from CNN import CNN
from plotter import Plotter

#variant2
from Preprocesser import Preprocessor
from TimeSeriesData_prep_dataset import TimeSeriesData_prep_dataset
from Dataloader_prep_dataset import Dataloader_prep_dataset


def main():
    #unpack arguments for training
    train_params = sys.argv[1:]
    print(train_params)
    """
    features_of_interest = train_params[0:2]
    num_epochs = int(train_params[2])
    GAMMA = float(train_params[3])
    GAMMA_reduction = float(train_params[4])
    num_pool = int(train_params[5])
    MMD_layer_activation_flag = train_params[6:]
    
    """
    experiment_name = str(train_params[0])
    num_epochs = int(train_params[1])
    GAMMA = float(train_params[2])
    GAMMA_reduction = float(train_params[3])
    num_pool = int(train_params[4])
    MMD_layer_activation_flag = train_params[5:11]
    features_of_interest = train_params[11:]
    print(MMD_layer_activation_flag)
    print(features_of_interest)
    
    MMD_layer_activation_flag = [eval(item.title()) for item in MMD_layer_activation_flag]
    
    
    #print(f"Features of interest: {features_of_interest} Num of epochs: {num_epochs} GAMMA: {GAMMA} num_pool: {num_pool}" )
    
    # check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"the device for executing the code is: {device}")

    #create random seeds
    random_seed = random.randrange(0,100)

    #Folder name to store data for each experiment
    features_of_interest_folder = features_of_interest[0].replace("/", "_")
    date =  datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    folder_to_store_data = "experiments/feature=" + str(features_of_interest_folder)  + "_" + "GAMMA=" + str(GAMMA) + "_" +"GAMMA_reduction" + str(GAMMA_reduction) + "_" + "num_pool=" + str(num_pool) + "_" + str(MMD_layer_activation_flag) + "_" + date

    #Generate folder structure to store plots and data
    current_directory = os.getcwd()
    path_learning_curve = os.path.join(current_directory, folder_to_store_data, "learning_curve")
    path_learning_curve_data = os.path.join(current_directory, folder_to_store_data, "learning_curve_data")
    path_data_distribution = os.path.join(current_directory, folder_to_store_data, "data_distribution")
    path_data_distribution_data = os.path.join(current_directory, folder_to_store_data, "data_distribution_data")
    path_accuracy = os.path.join(current_directory, folder_to_store_data, "accuracy")
    path_best_model = os.path.join(current_directory, folder_to_store_data, "best_model")

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
    if not os.path.exists(path_best_model): #Folder to store Accuracies of Training
        os.makedirs(path_best_model)


    #################
    #   Training    #
    #################


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
    writer_source_val = SummaryWriter(f'runs/{experiment_name}/source_val')
    writer_source_mmd = SummaryWriter(f'runs/{experiment_name}/source_mmd')
    writer_source_ce = SummaryWriter(f'runs/{experiment_name}/source_ce')
    writer_target_val = SummaryWriter(f'runs//{experiment_name}target_val')
    writer_target_mmd = SummaryWriter(f'runs/{experiment_name}/target_mmd')
    writer_target_ce = SummaryWriter(f'runs/{experiment_name}/target_ce')

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
    #list_of_source_BSD_states = ["2", "3", "11", "12", "20", "21"]
    #list_of_target_BSD_states = ["5", "6", "14", "15", "23", "24"]
    
    list_of_source_BSD_states = ["1", "2", "3", "4", "10", "11", "12", "13", "19", "20", "21", "22"]
    list_of_target_BSD_states = ["5", "6", "7", "9", "14", "15", "16", "18", "23", "24", "25", "27"]

    # Path where dataset is stored
    data_path = Path(os.getcwd()).parents[1]
    data_path = os.path.join(data_path, "data")

    # Dataloader
    dataloader_split_ce = 0.4
    dataloader_split_mmd = 0.2
    dataloader_split_val = 0.2

    batch_size = 32
    """
    ###Variant 2 ###
    source_numpy_array_names = ["source_X", "source_y"]
    target_numpy_array_names = ["target_X", "target_y"]
    preprocessor_source = Preprocessor(data_path, list_of_source_BSD_states, window_size, overlap_size, features_of_interest, source_numpy_array_names)
    preprocessor_target = Preprocessor(data_path, list_of_target_BSD_states, window_size, overlap_size, features_of_interest, target_numpy_array_names)
    features  = preprocessor_source.concatenate_data_from_BSD_state()
    _ = preprocessor_target.concatenate_data_from_BSD_state()
    dataset_source = TimeSeriesData_prep_dataset(data_path, window_size, overlap_size, source_numpy_array_names, features, features_of_interest)
    dataset_target = TimeSeriesData_prep_dataset(data_path, window_size, overlap_size, target_numpy_array_names, features, features_of_interest)
    dataloader_source = Dataloader_prep_dataset(dataset_source, dataloader_split_ce, dataloader_split_mmd, dataloader_split_val, batch_size, random_seed)
    dataloader_target = Dataloader_prep_dataset(dataset_target, dataloader_split_ce, dataloader_split_mmd, dataloader_split_val, batch_size, random_seed)
    source_loader = dataloader_source.create_dataloader()
    target_loader = dataloader_target.create_dataloader()
    """
    ###Variant 1####
    
    dataloader_source = Dataloader(data_path, list_of_source_BSD_states, window_size, overlap_size, features_of_interest, dataloader_split_ce, dataloader_split_mmd, dataloader_split_val, batch_size, random_seed)
    dataloader_target = Dataloader(data_path, list_of_target_BSD_states, window_size, overlap_size, features_of_interest, dataloader_split_ce, dataloader_split_mmd, dataloader_split_val, batch_size, random_seed)
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
    input_size = len(features_of_interest)
    hidden_fc_size_1 = 50
    hidden_fc_size_2 = 3
    output_size = 2
    model_cnn = CNN(input_size, hidden_fc_size_1, num_pool, window_size, random_seed)
    model_fc = Classifier(hidden_fc_size_1, hidden_fc_size_2, output_size, random_seed)

    #models to gpu if available
    model_cnn = model_cnn.to(device)
    model_fc = model_fc.to(device)
    
    if (next(model_cnn.parameters()).is_cuda and next(model_fc.parameters()).is_cuda):
        print("Models are on GPU!!")

    # Define Loss
    criterion = torch.nn.CrossEntropyLoss()
    MMD_loss_calculator = MMD_loss(fix_sigma = SIGMA)
    MMD_loss_CNN_calculator = MMD_loss_CNN(fix_sigma = SIGMA)
    loss_cnn = Loss_CNN(model_cnn, model_fc, criterion, MMD_loss_calculator, MMD_loss_CNN_calculator, MMD_layer_activation_flag)

    #Optimizer
    optimizer1 = torch.optim.Adam([
    {'params': model_cnn.parameters()},
    {'params': model_fc.parameters(), 'lr': 1e-4}
    ], lr=1e-2, betas=(0.9, 0.999))

    optimizer2 = torch.optim.Adam(model_fc.parameters(), lr=1e-2, betas=(0.9, 0.999))

    #Safe the random seed as txt file
    f_random_seed = open(f'{folder_to_store_data}/best_model/random_seed.txt', 'w')
    f_random_seed.write(str(random_seed))
    f_random_seed.close()

    #Safe the Model hyperparameter as txt file
    f_hyperparameter = open(f'{folder_to_store_data}/best_model/hyperparameter.txt', 'w')
    f_hyperparameter.write(f'features of interest: {features_of_interest}\n')
    f_hyperparameter.write(f'num_epochs: {num_epochs}\n')
    f_hyperparameter.write(f'GAMMA: {GAMMA}\n')
    f_hyperparameter.write(f'GAMMA_reduction: {GAMMA_reduction}\n')
    f_hyperparameter.write(f'num_pool: {num_pool}\n')
    f_hyperparameter.write(f'MMD_layer_flag: {MMD_layer_activation_flag}\n')
    f_hyperparameter.write(f'list_of_source_BSD_states: {list_of_source_BSD_states}\n')
    f_hyperparameter.write(f'list_of_target_BSD_states: {list_of_target_BSD_states}\n')
    f_hyperparameter.write(f'dataloader_split_ce: {dataloader_split_ce}\n')
    f_hyperparameter.write(f'dataloader_split_mmd: {dataloader_split_mmd}\n')
    f_hyperparameter.write(f'dataloader_split_val: {dataloader_split_val}\n')
    f_hyperparameter.write(f'batch_size: {batch_size}\n')
    f_hyperparameter.write(f'input_size_CNN: {input_size}\n')
    f_hyperparameter.write(f'hidden_fc_size_1: {hidden_fc_size_1}\n')
    f_hyperparameter.write(f'hidden_fc_size_2: {hidden_fc_size_2}\n')
    f_hyperparameter.write(f'output_size_FC: {output_size}\n')
    f_hyperparameter.write(f'SIGMA: {SIGMA}\n')
    f_hyperparameter.write(f'criterion: {criterion}\n')
    f_hyperparameter.write(f'optimizer1: {optimizer1}\n')
    f_hyperparameter.write(f'optimizer2: {optimizer2}\n\n')
    f_hyperparameter.write(f'Model CNN: {model_cnn}\n\n')
    f_hyperparameter.write(f'Model FC: {model_fc}') 
    f_hyperparameter.close()



    # Init variables which collect loss, accuracies for each epoch and train phase
    source_ce_loss_collected = 0
    target_ce_loss_collected = 0
    mmd_loss_collected = 0
    acc_total_source_collected = 0
    acc_total_target_collected = 0
    balanced_target_accuracy_collected = 0

    #store data about best performing model (balanced accuracy on validation set)
    max_target_val_accuracy = 0
    best_GAMMA = None
    best_features_of_interest = None
    best_pool = None


    # Train and Validate the model
    for epoch in range(num_epochs):

        GAMMA*=GAMMA_reduction
        print(GAMMA)

        #init array which collects the data in FC for plottnig the data distribution
        class_0_source_fc2_collect = torch.empty((0,3))
        class_1_source_fc2_collect = torch.empty((0,3))
        class_0_target_fc2_collect = torch.empty((0,3))
        class_1_target_fc2_collect = torch.empty((0,3))

        #init/reset list to collect data which is stored in one csv file line
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

                batch_data = batch_data.to(device)
                
                if batch_data.is_cuda and labels_source.is_cuda and labels_target.is_cuda:
                    print("Samples are all on GPU !!")

                #validation
                if phase == "val":
                    model_cnn.train(False)
                    model_fc.train(False)
                    
                    with torch.no_grad():
                        _, mmd_loss, source_ce_loss, target_ce_loss, acc_total_source, acc_total_target, balanced_target_accuracy, class_0_source_fc2, class_1_source_fc2, class_0_target_fc2, class_1_target_fc2 = loss_cnn.forward(batch_data, labels_source, labels_target, MMD_loss_flag_phase[phase], GAMMA)
                        
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
                    loss, mmd_loss, source_ce_loss, target_ce_loss, acc_total_source, acc_total_target, _, _, _, _, _ = loss_cnn.forward(batch_data, labels_source, labels_target, MMD_loss_flag_phase[phase], GAMMA)
                    
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
                mmd_loss_collected += mmd_loss.item()
                source_ce_loss_collected += source_ce_loss.item()
                target_ce_loss_collected += target_ce_loss.item()
                acc_total_source_collected += acc_total_source
                acc_total_target_collected += acc_total_target
                balanced_target_accuracy_collected += balanced_target_accuracy
            
            # store data distribution in latent feature space fc2 in csv
            if phase == "val" and (epoch ==0 or epoch ==20 or epoch == 40 or epoch ==80):
                
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
            running_balanced_target_accuracy = balanced_target_accuracy_collected / len(target_loader[phase])

            #In each epoch check the average target accuracy of the model on the validation set and store model if it performed better than in the previous epochs
            if phase == "val":
                if max_target_val_accuracy < running_balanced_target_accuracy:
                    max_target_val_accuracy = running_balanced_target_accuracy
                    print(running_balanced_target_accuracy)
                    torch.save(model_cnn.state_dict(), f'{folder_to_store_data}/best_model/model_cnn.pt')
                    torch.save(model_fc.state_dict(), f'{folder_to_store_data}/best_model/model_fc.pt')
                    best_GAMMA = GAMMA
                    best_features_of_interest = features_of_interest
                    best_pool = num_pool

            #Add train data to tensorboard list
            writer_source[phase].add_scalar(f'accuracy', running_acc_source, epoch)
            writer_target[phase].add_scalar(f'accuracy', running_acc_target, epoch)
            writer_source[phase].add_scalar(f'ce_loss', running_source_ce_loss, epoch)
            writer_target[phase].add_scalar(f'ce_loss', running_target_ce_loss, epoch)
            writer_source[phase].add_scalar(f'mmd_loss', running_mmd_loss, epoch)

            #collect data which is stored in one line of csv
            learning_curve_data_collect = learning_curve_data_collect + [running_acc_source, running_acc_target, running_source_ce_loss, running_target_ce_loss, running_mmd_loss]
            f_accuracy_collect = f_accuracy_collect + [running_acc_source, running_acc_target]

            #Reset variable for collected loss, accuracies for each epoch and train phase
            source_ce_loss_collected = 0
            target_ce_loss_collected = 0
            mmd_loss_collected = 0
            acc_total_source_collected = 0
            acc_total_target_collected = 0
            balanced_target_accuracy_collected = 0

        #store write one line in csv 
        f_learning_curve_writer.writerow(learning_curve_data_collect)
        f_accuracy_writer.writerow(f_accuracy_collect)
        
        print(f"Epoch {epoch+1}/{num_epochs} successfull")

    #close csv writer for accuracy and learning curves
    f_accuracy.close()
    f_learning_curve.close()

    #plot learning curves and data distribtuion from csv files
    plotter.plot_distribution()
    plotter.plot_curves()

    print(f"With an Accuracy of: {max_target_val_accuracy} the model with the following hyperparameter performed best:\nbest_features_of_interest: {best_features_of_interest}\nbest_GAMMA: {best_GAMMA}\nbest_pool: {best_pool}")

    ################
    #   Testing    #
    ################

    model_cnn_test =  CNN(input_size, hidden_fc_size_1, num_pool, window_size, random_seed)
    model_cnn_test.load_state_dict(torch.load(f'{folder_to_store_data}/best_model/model_cnn.pt'))
    model_cnn_test.eval()

    model_fc_test = Classifier(hidden_fc_size_1, hidden_fc_size_2, output_size, random_seed)
    model_fc_test.load_state_dict(torch.load(f'{folder_to_store_data}/best_model/model_fc.pt'))
    model_fc_test.eval() 

    acc_total_target_test_collected = 0

    test_loader_target = iter(target_loader["test"])
    #iterate through batches of phase specific dataloader
    for _ in range(len(test_loader_target)):
                
        ########Forward pass########
        batch_data_target_test, labels_target_test = test_loader_target.next() #batch_size number of windows from target domain
        _, _, _, _, x_fc1_test = model_cnn_test(batch_data_target_test.float())
        _, x_fc3_test = model_fc_test(x_fc1_test)

        #Accuracy Test
        argmax_target_test_pred = torch.argmax(x_fc3_test[:batch_size, :], dim=1)
        result_target_test_pred = argmax_target_test_pred == labels_target_test
        correct_target_test_pred = result_target_test_pred[result_target_test_pred == True]
        acc_total_target_test = 100 * len(correct_target_test_pred)/len(labels_target_test)
        acc_total_target_test_collected += acc_total_target_test

    running_acc_target_test = acc_total_target_test_collected / len(test_loader_target)
    #Safe the random seed as txt file
    f_target_test_accuracy = open(f'{folder_to_store_data}/best_model/target_test_accuracy.txt', 'w')
    f_target_test_accuracy.write(str(running_acc_target_test))
    f_target_test_accuracy.close()




if __name__ == "__main__":
    main()
