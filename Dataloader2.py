import os
import csv
from venv import create
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, Subset
import torch


from skimage.util.shape import view_as_windows
import warnings
import matplotlib.pyplot as plt


def regime_separation(data, vel_profile_feature, window_size, overlap_size, vel_cut_off_value):
    """
    Create physically usefull windows. Window should contain a physically meaningful state of the machine. In this case we cut out windows with constant velocity.

    INPUT: 
    @data: data numpy array of shape [elements per file, features]
    @vel_profile: defines the windows with constant velocity which are used to window all the features of interest

    OUTPUT:
    @data: data numpy array of shape [number_of_windows, elements per window, features]
    """
    data_positiv_vel = data[vel_profile_feature>=abs(vel_cut_off_value)] #get all data for max velocity 
    data_negative_vel = data[vel_profile_feature<=-abs(vel_cut_off_value)] #get all data for min velocity 
    data_positiv_vel_splitted = split_data(data_positiv_vel, window_size, overlap_size) #split positiv velocity data into windows
    data_negative_vel_splitted = split_data(data_negative_vel, window_size, overlap_size) #split positiv velocity data into windows
    #num_windows = min(np.shape(data_positiv_vel_splitted)[0], np.shape(data_negative_vel_splitted)[0]) # get samller number of windows
    #data_positiv_vel_splitted = data_positiv_vel_splitted[:num_windows, :, :] #slice positiv velocity data to samller number of windows
    #data_negative_vel_splitted = data_negative_vel_splitted[:num_windows, :, :] #slice negative velocity data to samller number of windows

    data = np.concatenate((data_positiv_vel_splitted, data_negative_vel_splitted), axis = 0) # concatenate along axis 1 to treat the features of interests separatelly for positiv and negativ velocity
    return data





def split_data(data, window_size, overlap_size):
    """
    Split data in windows of equal size with overlap
    
    INPUT:
    @data: data numpy array of shape [elements per file, features]
    @window: number of elements per window
    @overlap_size: defines the overlapping elements between consecutive windows
    
    OUTPUT
    @data: data numpy array of shape [number_of_windows, elements per window, features]
    """

    
    if window_size> np.shape(data)[0]:
        raise Exception(f"Window is too big for data. Window size must at maximum be: {np.shape(data)[0]}")

    if window_size==overlap_size:
        plt.plot(data[:,4])
        plt.show()
        raise Exception("Overlap arg must be smaller than length of windows")
    S = window_size - overlap_size
    nd0 = ((len(data)-window_size)//S)+1
    if nd0*S-S!=len(data)-window_size:
        warnings.warn("Not all elements were covered")
    return view_as_windows(data, (window_size,data.shape[1]), step=S)[:,0,:,:]


def del_nan_element(data_with_nan):
    """
    Delete all elements in the data which have any nan valued feature
    
    INPUT:
    @data_with_nan: data numpy array containing nan_values
    
    OUTPUT
    @data_with_nan: data numpy array inlcuding just elements per window which do have no nan_vaues in any feature
    """
    nan_val = np.isnan(data_with_nan) #mask for all nan_elements as 2d array [elements_per_window, features]
    nan_val = np.any(nan_val,axis = 1) #mask for all nan_rows as 1d array [elements_per_window]
    return data_with_nan[nan_val==False]


def create_folder_dictionary(list_of_BSD_states, data_path):
    """
    Create a dictionaty for testing and training containing folder names as keys and files as values
    
    INPUT:
    @list_of_train_BSD_states: list containing the training BSD states as string
    @list_of_test_BSD_states: list containing the testing BSD states as string
    @data_path: data directory containing folders for each BSD state
    
    OUTPUT
    @training_folders: dictionary folders and keys for training
    @testing_folders: dictionary folders and keys for testing
    """
    
    data_path = data_path
    state_dictionary = {
        "1":"NR01_20200317_PGS_31_BSD_31",
        "2":"NR02_20200423_PGS_31_BSD_21",
        "3":"NR03_20200424_PGS_31_BSD_11",
        "4":"NR04_20200424_PGS_31_BSD_P1",
        "5":"NR05_20200930_PGS_31_BSD_22",
        "6":"NR06_20201001_PGS_31_BSD_12",
        "7":"NR07_20201001_PGS_31_BSD_32",
        "8":"NR08_20200918_PGS_31_BSD_33",
        "9":"NR09_20200917_PGS_31_BSD_P2",
        "10":"NR10_20200502_PGS_21_BSD_31",
        "11":"NR11_20200429_PGS_21_BSD_21",
        "12":"NR12_20200429_PGS_21_BSD_11",
        "13":"NR13_20200428_PGS_21_BSD_P1",
        "14":"NR14_20200731_PGS_21_BSD_22",
        "15":"NR15_20200901_PGS_21_BSD_12",
        "16":"NR16_20200908_PGS_21_BSD_32",
        "17":"NR17_20200717_PGS_21_BSD_33",
        "18":"NR18_20200714_PGS_21_BSD_P2",
        "19":"NR19_20200505_PGS_11_BSD_31",
        "20":"NR20_20200507_PGS_11_BSD_21",
        "21":"NR21_20200508_PGS_11_BSD_11",
        "22":"NR22_20200508_PGS_11_BSD_P1",
        "23":"NR23_20200511_PGS_11_BSD_22",
        "24":"NR24_20200512_PGS_11_BSD_12",
        "25":"NR25_20200512_PGS_11_BSD_32",
        "26":"NR26_20200513_PGS_11_BSD_33",
        "27":"NR27_20200513_PGS_11_BSD_P2",
    }
    
    
    data_folders = {}
    
    for element in list_of_BSD_states:
        data_folders[state_dictionary[element]]=os.listdir(os.path.join(data_path,state_dictionary[element]))
    return data_folders





def get_features(path):
    """
    Creates a list of all feature names
    INPUT:
    @path: path to any BSD file since the features are the same for all files
    
    OUTPUT
    @features: list of features:
    ['C:s_ist/X', 'C:s_soll/X', 'C:s_diff/X', 'C:v_(n_ist)/X', 'C:v_(n_soll)/X', 'C:P_mech./X', 'C:Pos._Diff./X',
    'C:I_ist/X', 'C:I_soll/X', 'C:x_bottom', 'C:y_bottom', 'C:z_bottom', 'C:x_nut', 'C:y_nut', 'C:z_nut',
    'C:x_top', 'C:y_top', 'C:z_top', 'D:s_ist/X', 'D:s_soll/X', 'D:s_diff/X', 'D:v_(n_ist)/X', 'D:v_(n_soll)/X',
    'D:P_mech./X', 'D:Pos._Diff./X', 'D:I_ist/X', 'D:I_soll/X', 'D:x_bottom', 'D:y_bottom', 'D:z_bottom',
    'D:x_nut', 'D:y_nut', 'D:z_nut', 'D:x_top', 'D:y_top', 'D:z_top', 'S:x_bottom', 'S:y_bottom', 'S:z_bottom',
    'S:x_nut', 'S:y_nut', 'S:z_nut', 'S:x_top', 'S:y_top', 'S:z_top', 'S:Nominal_rotational_speed[rad/s]',
    'S:Actual_rotational_speed[µm/s]', 'S:Actual_position_of_the_position_encoder(dy/dt)[µm/s]',
    'S:Actual_position_of_the_motor_encoder(dy/dt)[µm/s]']
    """
    
    with open(path, 'r') as file:
        csvreader = csv.reader(file)
        features = next(csvreader)
    return features



def concatenate_data_from_BSD_state(data_folders, data_path, features_of_interest, window_size, overlap_size, vel_cut_off_value):
    """
    Concatenates all the windowed data from each file to one big torch array
    INPUT:
    @folders: dictionary containing folders (as keys) and files (as values) to downloaded
    @data_path: data directory containing folders for each BSD state
    @features_of_interest: list of features which should be included for training
    @window_size: number of elements per widow
    
    OUTPUT:
    @n_samples: number of total elements from all included files
    @x_data: torch array containing all the data elements 
    @y_data: torch array containing the labels for all elements
    """
    
    
    # arrays to collect data and label
    x_data_concatenated = None
    y_data_concatenated = None
    
    
    iterator = 0
    first = True
    
    
    for BSD_path in data_folders.keys(): #folder path
        for file_path in data_folders[BSD_path]: #file path 
            path_BSD_file = os.path.join(data_path, BSD_path, file_path) # concatenate the data_path, folder and file path
            #in first iteration get a list if all features
            if first == True:
                features = get_features(path_BSD_file)
            data_BSD_file_all = np.genfromtxt(path_BSD_file, dtype = np.dtype('d'), delimiter=',')[1:,:] #write csv in numpy
            feature_index_list = np.where(np.isin(features, features_of_interest)) #get index for all features of interest
            data_BSD_file = data_BSD_file_all[:,feature_index_list] #slice numpy array such that just features of interest are included
            data_BSD_file = np.squeeze(data_BSD_file, axis = 1) # one unnecessary extra dimension was created while slicing
            index_vel_profile_feature = features.index('C:v_(n_ist)/X') #get index for velocity feature which is used for windowing
            vel_profile_feature = data_BSD_file_all[:,index_vel_profile_feature] #get velocity feature vector which is used for windowing

            concatenate_data_BSD_and_vel_profile = np.concatenate((data_BSD_file, np.expand_dims(vel_profile_feature,axis = 1)), axis = 1) #concatenate data and velocity feature to commonly delete nan values and generating feature vectors of equal length
            concatenate_data_BSD_and_vel_profile_del_nan = del_nan_element(concatenate_data_BSD_and_vel_profile) #delete all elements with any nan feature
            vel_profile_feature = concatenate_data_BSD_and_vel_profile_del_nan[:,-1] #get velocity vector with deleted nan
            data_BSD_file = concatenate_data_BSD_and_vel_profile_del_nan[:,:-1] #get data with deleted nan 
            data_BSD_file = regime_separation(data_BSD_file, vel_profile_feature, window_size, overlap_size, vel_cut_off_value) #generate windows
            data_BSD_file = np.swapaxes(data_BSD_file,1,2) #swap axes for CNN
            
            
            #rewrite labels as BSD_condition_1 = 0, BSD_condition_2 = 1, BSD_condition_3 = 2, BSD_condition_P1 = 3
            label = BSD_path[-2] #take the first number of the BSD state for class label
            if label == "P":
                label = int(3)
            else:
                label =int(int(label)-1)
            
            
            
            #concatenate the data from each file in one numpy array
            if  first == True: #overwrite variable
                x_data_concatenated = np.copy(data_BSD_file)
                y_data_concatenated = np.copy(np.asarray([label]*np.shape(data_BSD_file)[0]))
                first = False
            else: #concatenate data numpy arrays
                x_data_concatenated = np.concatenate((x_data_concatenated, data_BSD_file), axis=0)
                y_data_concatenated = np.concatenate((y_data_concatenated,np.asarray([label]*np.shape(data_BSD_file)[0])), axis=0)
            
            iterator +=1
            print(f"{iterator}/{len(data_folders.keys())*len(data_folders[list(data_folders.keys())[0]])} folders downloaded")
            print(f"downloaded folder: {BSD_path}/{file_path}")
            print(f"Shape of collected datafram: X_shape: {np.shape(x_data_concatenated)}, Y_shape: {np.shape(y_data_concatenated)}")
    
    #generate torch array
    n_samples = np.shape(x_data_concatenated)[0]
    x_data = torch.from_numpy(x_data_concatenated)
    y_data = torch.from_numpy(y_data_concatenated)
    
    return n_samples, x_data, y_data




class TimeSeriesData(Dataset):
    """
    Class for creating dataset using PyTorch data primitive Dataset. An instance of this class can be used in the 
    PyTorch data primitive Dataloader
    
    The following patameters can be adjusted:
    @windwo_size: Size of window which is used as Input in CNN
    @feature_of_interest: List of all features which should be used in the CNN
    @list_of_train_BSD_states: List of BSD states which should be used for training. Be careful at least 4 BSD
    states representing the 4 different classes should be included for the training
    @list_of_test_BSD_states: List of BSD states which should be used for testing
    """
    
    
    def __init__(self, data_path, list_of_BSD_states, window_size, overlap_size, vel_cut_off_value, features_of_interest):
        
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.features_of_interest = features_of_interest
        self.list_of_BSD_states =list_of_BSD_states
        

        
        data_folders = create_folder_dictionary(self.list_of_BSD_states, data_path)
        self.n_samples, self.x_data, self.y_data = concatenate_data_from_BSD_state(data_folders, data_path, features_of_interest, window_size, overlap_size ,vel_cut_off_value)
        
                  
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


def create_dataloader(data_path, list_of_BSD_states, window_size, overlap_size, vel_cut_off_value, features_of_interest, dataloader_type, dataloader_split, batch_size):
    """
    Create dataloader for training, validation, testing
    
    INPUT:
    @dataset_train: dataset with samples for training and validation from training domain
    @dataset_test: dataset with samples for testing from testing domain

    OUTPUT
    @data_loader: dictionary which contains the dataloader for training and val. Dataloaders can be accesses with keys "train", "val"
    @test_loader: dataloader for testing

    """

    #get dataset
    dataset = TimeSeriesData(data_path, list_of_BSD_states, window_size, overlap_size, vel_cut_off_value, features_of_interest)

    # define train/val dimensions
    if dataloader_type == "train":
        train_size = int(dataloader_split * len(dataset))
        validation_size = len(dataset) - train_size

        #split dataset randomly
        training_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

        #dataloader
        train_loader = DataLoader(dataset=training_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=2)


        val_loader = DataLoader(dataset=validation_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=2)

        data_loader = {}
        data_loader['train'] = train_loader
        data_loader['val'] = val_loader


    else:
        test_size = int(dataloader_split * len(dataset))
        validation_size = len(dataset) - test_size

        #split dataset randomly
        test_dataset, validation_dataset = torch.utils.data.random_split(dataset, [test_size, validation_size])

        #dataloader
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=2)


        val_loader = DataLoader(dataset=validation_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=2)

        data_loader = {}
        data_loader['test'] = test_loader
        data_loader['val'] = val_loader


    return data_loader



if __name__ == '__main__':
    from pathlib import Path
    window_size = 1024
    overlap_size = 0
    vel_cut_off_value = 23600
    features_of_interest = ['C:s_ist/X', 'C:s_soll/X', 'C:s_diff/X', 'C:v_(n_ist)/X', 'C:v_(n_soll)/X', 'C:P_mech./X', 'C:Pos._Diff./X',
    'C:I_ist/X', 'C:I_soll/X', 'C:x_bottom', 'C:y_bottom', 'C:z_bottom', 'C:x_nut', 'C:y_nut', 'C:z_nut',
    'C:x_top', 'C:y_top', 'C:z_top', 'D:s_ist/X', 'D:s_soll/X', 'D:s_diff/X', 'D:v_(n_ist)/X', 'D:v_(n_soll)/X',
    'D:P_mech./X', 'D:Pos._Diff./X', 'D:I_ist/X', 'D:I_soll/X', 'D:x_bottom', 'D:y_bottom', 'D:z_bottom',
    'D:x_nut', 'D:y_nut', 'D:z_nut', 'D:x_top', 'D:y_top', 'D:z_top']
    list_of_BSD_states = ["1"]#, "2", "3", "4", "10", "11", "12", "13", "19", "20", "21", "22"]
    data_path = Path(os.getcwd()).parents[0]
    data_path = os.path.join(data_path, "data")

    #dataset = TimeSeriesData(data_path, list_of_BSD_states, window_size, overlap_size, vel_cut_off_value, features_of_interest)

    dataloader_split = 0.9
    batch_size = 4
    dataloader = create_dataloader(data_path, list_of_BSD_states, window_size, overlap_size, vel_cut_off_value, features_of_interest, "test", dataloader_split, batch_size)
    for i, (window, labels) in enumerate(dataloader["val"]):
        print(window)



