import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

class Plotter():
    def __init__(self, datapath):
        self.datapath = datapath

    def plot_distribution(self):
        #train_params = sys.argv[1:]
        #csv_file_path = train_params[0]
        #csv_file_path = os.path.join(self.datapath, "data_distribution_data", "data_distribution_0.csv")
        csv_file_path_folder = os.path.join(self.datapath, "data_distribution_data")
        csv_file_path_folder_elements = os.listdir(csv_file_path_folder)
        for element in csv_file_path_folder_elements:
            csv_file_path = os.path.join(self.datapath, "data_distribution_data", element)
            df = pd.read_csv(csv_file_path)
            fig = plt.figure()
            plt.gcf().set_size_inches((20, 20)) 
            ax = fig.add_subplot(projection='3d')

            #plot data
            m = [1,2,3,4]
            columns = df.columns.values.tolist()
            for i in range(4):
                ax.scatter(df[columns[0+i*3]], df[columns[1+i*3]], df[columns[2+i*3]], marker=m[i])
            
            #label axis
            ax.set_xlabel('Neuron 1 $\longrightarrow$', rotation=0, labelpad=10, size=20)
            ax.set_ylabel('Neuron 2 $\longrightarrow$', rotation=0, labelpad=10, size=20)
            ax.set_zlabel('Neuron 3 $\longrightarrow$', rotation=0, labelpad=10, size=20)
            plt.rcParams.update({'font.size': 10})
                        
            #show and safe fig
            #fig.savefig(f"{sys.argv[1][:-3]}pdf", format='pdf')  
            print(f"{self.datapath}/data_distribution/{element}.pdf")    
            fig.savefig(f"{self.datapath}/data_distribution/{element[:-4]}.pdf", format='pdf')     

    def plot_curves(self):
        csv_file_path = os.path.join(self.datapath, "plots_data", "plots.csv")
        df = pd.read_csv(f'{csv_file_path}')

        #Plot training curves
        fig1 = plt.figure()
        plt.title('Accuracy Source Domain')
        plt.plot(df["running_acc_source_ce"], 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_acc_source_mmd"], 'ro-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_acc_source_val"], 'go-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$")
        plt.ylabel("Accuracy Source Domain $\longrightarrow$")
        plt.legend()
        fig1.savefig(f"{self.datapath}/plots/Accuracy_Source_Domain.pdf", format='pdf')

        fig2 = plt.figure()
        plt.title('Accuracy Target Domain')
        plt.plot(df["running_acc_target_ce"], 'co-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_acc_target_mmd"], 'mo-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_acc_target_val"], 'yo-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$")
        plt.ylabel("Accuracy Target Domain $\longrightarrow$")
        plt.legend()
        fig2.savefig(f"{self.datapath}/plots/Accuracy_Target_Domain.pdf", format='pdf')

        fig3 = plt.figure()
        plt.title('CE-Loss Source Domain')
        plt.plot(df["running_source_ce_loss_ce"], 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_source_ce_loss_mmd"], 'ro-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_source_ce_loss_val"], 'go-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$")
        plt.ylabel("CE-Loss Source Domain $\longrightarrow$")
        plt.legend()
        fig3.savefig(f"{self.datapath}/plots/CE_Loss_Source_Domain.pdf", format='pdf')

        fig4 = plt.figure()
        plt.title('CE-Loss Target Domain')
        plt.plot(df["running_target_ce_loss_ce"], 'co-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_target_ce_loss_mmd"], 'mo-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_target_ce_loss_val"], 'yo-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$")
        plt.ylabel("CE-Loss Target Domain $\longrightarrow$")
        plt.legend()
        fig4.savefig(f"{self.datapath}/plots/CE_Loss_Target_Domain.pdf", format='pdf')

        fig5 = plt.figure()
        plt.title('MMD-Loss')
        plt.plot(df["running_mmd_loss_ce"], 'bo-', label = 'CE-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_mmd_loss_mmd"], 'ro-', label = 'MMD-Loss', linewidth=1,markersize=0.1)
        plt.plot(df["running_mmd_loss_val"], 'go-', label = 'Val', linewidth=1,markersize=0.1)
        plt.xlabel("Epoch $\longrightarrow$")
        plt.ylabel("MMD-Loss $\longrightarrow$")
        plt.legend()
        fig5.savefig(f"{self.datapath}/plots/MMD_Loss.pdf", format='pdf')