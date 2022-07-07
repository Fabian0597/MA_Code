import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pywt
from skimage.transform import resize
import time

from Preprocesser import Preprocessor
from TimeSeriesData_prep_dataset import TimeSeriesData_prep_dataset
from Dataloader_prep_dataset import Dataloader_prep_dataset


def wavelet_transform(data):
    scales = np.arange(1, 65) # range of scales
    rescale_size = scales[-1]
    waveletname = 'morl'
    X_cwt = np.ndarray(shape=(np.shape(data)[0], np.shape(data)[1], rescale_size, rescale_size), dtype = 'float32')

    for i, sample in enumerate(range(np.shape(data)[0])):
        time_start = time.time()
        for j, feature in enumerate(range(np.shape(data)[1])):
            signal = data[sample, feature, :]
            coeffs, freqs = pywt.cwt(signal, scales, waveletname, 1) #(64,1024) (64)
            rescale_coeffs = resize(coeffs, (rescale_size, rescale_size), mode = 'constant')
            X_cwt[i,j,:,:] = rescale_coeffs
        time_end = time.time()
        time_diff_s = (time_end-time_start)*(np.shape(data)[0]-i)
        print(f"{i}/{np.shape(data)[0]}")
        print(f"estimated time: {int(time_diff_s//60)}:{int(time_diff_s%60)}min")
    return X_cwt

def load_data():
    data_path = Path(os.getcwd()).parents[1]
    data_path = os.path.join(data_path, "data")
    source_numpy_array_names = ["source_X", "source_y"]
    target_numpy_array_names = ["target_X", "target_y"]
    window_size = 1024
    overlap_size = 0

    list_of_source_BSD_states = ["2", "3", "11", "12", "20", "21"]
    list_of_target_BSD_states = ["5", "6", "14", "15", "23", "24"]


    features_of_interest=        ['C:s_ist/X', 'C:s_soll/X', 'C:s_diff/X', 'C:v_(n_ist)/X', 'C:v_(n_soll)/X', 'C:P_mech./X', 'C:Pos_Diff./X', 'C:I_ist/X', 'C:I_soll/X', 'C:x_bottom', 'C:y_bottom', 'C:z_bottom', 'C:x_nut', 'C:y_nut', 'C:z_nut',
            'C:x_top', 'C:y_top', 'C:z_top', 'D:s_ist/X', 'D:s_soll/X', 'D:s_diff/X', 'D:v_(n_ist)/X', 'D:v_(n_soll)/X',
            'D:P_mech./X', 'D:Pos._Diff./X', 'D:I_ist/X', 'D:I_soll/X', 'D:x_bottom', 'D:y_bottom', 'D:z_bottom',
            'D:x_nut', 'D:y_nut', 'D:z_nut', 'D:x_top', 'D:y_top', 'D:z_top', 'S:x_bottom', 'S:y_bottom', 'S:z_bottom',
            'S:x_nut', 'S:y_nut', 'S:z_nut', 'S:x_top', 'S:y_top', 'S:z_top', 'S:Nominal_rotational_speed[rad/s]',
            'S:Actual_rotational_speed[µm/s]', 'S:Actual_position_of_the_position_encoder(dy/dt)[µm/s]',
            'S:Actual_position_of_the_motor_encoder(dy/dt)[µm/s]']



    source_numpy_array_names = ["source_X", "source_y"]
    target_numpy_array_names = ["target_X", "target_y"]
    preprocessor_source = Preprocessor(data_path, list_of_source_BSD_states, window_size, overlap_size, features_of_interest, source_numpy_array_names)
    preprocessor_target = Preprocessor(data_path, list_of_target_BSD_states, window_size, overlap_size, features_of_interest, target_numpy_array_names)
    features  = preprocessor_source.concatenate_data_from_BSD_state()
    _ = preprocessor_target.concatenate_data_from_BSD_state()
    dataset_source = TimeSeriesData_prep_dataset(data_path, window_size, overlap_size, source_numpy_array_names, features, features_of_interest)
    dataset_target = TimeSeriesData_prep_dataset(data_path, window_size, overlap_size, target_numpy_array_names, features, features_of_interest)

    X_data_source = dataset_source.x_data.numpy()
    y_data_source = dataset_source.y_data.numpy()

    X_data_target = dataset_target.x_data.numpy()
    y_data_target = dataset_target.y_data.numpy()



    X_data_source_cwt = wavelet_transform(X_data_source)
    X_data_target_cwt = wavelet_transform(X_data_target)

    data_path = Path(os.getcwd()).parents[1]
    data_path = os.path.join(data_path, "data")

    np.save(os.path.join(data_path, "X_data_source_cwt"), X_data_source_cwt)
    np.save(os.path.join(data_path, "X_data_target_cwt"), X_data_target_cwt)

def test_data():
    data_path = Path(os.getcwd()).parents[1]
    data_path = os.path.join(data_path, "data")
    X_data_source_cwt = np.load(os.path.join(data_path, "X_data_source_cwt.npy"))
    X_data_target_cwt = np.load(os.path.join(data_path, "X_data_target_cwt.npy"))
    print(np.shape(X_data_source_cwt))

    #plt.plot(X_data_source_cwt[0,0,:,:])
    #plt.show()
def test_wavelets():
    data_path = Path(os.getcwd()).parents[0]
    data_path = os.path.join(data_path, "DOMAIN_ADAPTION/wavelet_transforms_np")
    X_data_source_0_cwt = np.load(os.path.join(data_path, "wavelet_transforms_source_0.npy"))
    X_data_source_1_cwt = np.load(os.path.join(data_path, "wavelet_transforms_source_1.npy"))
    X_data_target_0_cwt = np.load(os.path.join(data_path, "wavelet_transforms_target_0.npy"))
    X_data_target_1_cwt = np.load(os.path.join(data_path, "wavelet_transforms_target_1.npy"))
    diff_target = X_data_target_1_cwt-X_data_target_0_cwt
    diff_source = X_data_source_1_cwt-X_data_source_0_cwt
    diff_class_0= X_data_target_0_cwt-X_data_source_0_cwt
    diff_class_1 = X_data_target_1_cwt-X_data_source_1_cwt
    print(np.sum(diff_target), np.sum(diff_source), np.sum(diff_class_0), np.sum(diff_class_1))
    

    features = ["C:s_ist/X", "C:s_soll/X", "C:s_diff/X", "C:v_(n_ist)/X", "C:v_(n_soll)/X", "C:P_mech./X", "C:Pos._Diff./X",
        "C:I_ist/X", "C:I_soll/X", "C:x_bottom", "C:y_bottom", "C:z_bottom", "C:x_nut", "C:y_nut", "C:z_nut",
        "C:x_top", "C:y_top", "C:z_top", "D:s_ist/X", "D:s_soll/X", "D:s_diff/X", "D:v_(n_ist)/X", "D:v_(n_soll)/X",
        "D:P_mech./X", "D:Pos._Diff./X", "D:I_ist/X", "D:I_soll/X", "D:x_bottom", "D:y_bottom", "D:z_bottom",
        "D:x_nut", "D:y_nut", "D:z_nut", "D:x_top", "D:y_top", "D:z_top", "S:x_bottom", "S:y_bottom", "S:z_bottom",
        "S:x_nut", "S:y_nut", "S:z_nut", "S:x_top", "S:y_top", "S:z_top", "S:Nominal_rotational_speed[rad/s]",
        "S:Actual_rotational_speed[µm/s]", "S:Actual_position_of_the_position_encoder(dy/dt)[µm/s]",
        "S:Actual_position_of_the_motor_encoder(dy/dt)[µm/s]"]

    for i in range(np.shape(X_data_target_0_cwt)[1]):
        plot_wavelets(diff_target[:,i,:,:], diff_source[:,i,:,:], diff_class_0[:,i,:,:], diff_class_1[:,i,:,:], features[i])

    """
    diff_source = np.sum(diff_source, axis=1)
    diff_source = np.sum(diff_source, axis=0)
    fig = plt.figure()
    im = plt.imshow(diff_source, interpolation='None')
    cbar = plt.colorbar(im)
    cbar.set_label("Difference $\longrightarrow$")
    ax = plt.gca()
    ax.grid(color='red', linestyle='-.', linewidth=1)
    plt.xlabel("Time $\longrightarrow$")
    plt.ylabel("Sales $\longrightarrow$")
    plt.title(f'Target_diff', fontsize=8)
    #plt.savefig(f"{data_path}/Source_Target_diff_Class_1.pdf", format='pdf')
    plt.show()
    """

def plot_wavelets(diff_target, diff_source, diff_class_0, diff_class_1, features):
    
    
    diff_target = np.sum(diff_target, axis=0)
    
    diff_source = np.sum(diff_source, axis=0)

    diff_class_0 = np.sum(diff_class_0, axis=0)

    diff_class_1 = np.sum(diff_class_1, axis=0)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(diff_target, interpolation='None')
    axs[0, 0].set_xlabel("Time $\longrightarrow$", fontsize=7)
    axs[0, 0].set_ylabel("Scales $\longrightarrow$", fontsize=7)
    axs[0, 0].set_title(f'Target_diff_feature_{features}', fontsize=7)
    axs[0, 1].imshow(diff_source, interpolation='None')
    axs[0, 1].set_xlabel("Time $\longrightarrow$", fontsize=7)
    axs[0, 1].set_ylabel("Scales $\longrightarrow$", fontsize=7)
    axs[0, 1].set_title(f'Source_diff_feature_{features}', fontsize=7)
    axs[1, 0].imshow(diff_class_0, interpolation='None')
    axs[1, 0].set_xlabel("Time $\longrightarrow$", fontsize=7)
    axs[1, 0].set_ylabel("Scales $\longrightarrow$", fontsize=7)
    axs[1, 0].set_title(f'Domain_class_0_diff_feature_{features}', fontsize=7)
    im4 = axs[1, 1].imshow(diff_class_1, interpolation='None')
    axs[1, 1].set_xlabel("Time $\longrightarrow$", fontsize=7)
    axs[1, 1].set_ylabel("Scales $\longrightarrow$", fontsize=7)
    axs[1, 1].set_title(f'Domain_class_1_diff_feature_{features}', fontsize=7)
    cbar4 = plt.colorbar(im4)
    cbar4.set_label("Difference $\longrightarrow$", fontsize=7)
    fig.tight_layout()

    features = features.replace("/", "_")

    #plt.show()
    plt.savefig(f"wavelet_features/wavelet_features_{features}.pdf", format='pdf')

test_wavelets()