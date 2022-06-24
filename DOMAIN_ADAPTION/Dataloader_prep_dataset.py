
from torch.utils.data import DataLoader
import torch
torch.manual_seed(0)
from TimeSeriesData import TimeSeriesData



class Dataloader_prep_dataset():
    def __init__(self, dataset, dataloader_split_ce, dataloader_split_mmd, dataloader_split_val, batch_size, random_seed) -> None:
        
        self.dataset = dataset
        self.dataloader_split_ce = dataloader_split_ce
        self.dataloader_split_mmd = dataloader_split_mmd
        self.dataloader_split_val = dataloader_split_val
        self.batch_size = batch_size

        torch.manual_seed(0)

    def create_dataloader(self):
        """
        Create dataloader for training, validation, testing
    
        INPUT:
        @dataset_train: dataset with samples for training and validation from training domain
        @dataset_test: dataset with samples for testing from testing domain

        OUTPUT
        @data_loader: dictionary which contains the dataloader for training and val. Dataloaders can be accesses with keys "train", "val"
        @test_loader: dataloader for testing

        """

        # define train/val dimensions
        dataset_size_ce= int(self.dataloader_split_ce * len(self.dataset))
        dataset_size_mmd= int(self.dataloader_split_mmd * len(self.dataset))
        dataset_size_val= int(self.dataloader_split_val * len(self.dataset))
        dataset_size_test = len(self.dataset) - dataset_size_ce - dataset_size_mmd - dataset_size_val

        #split dataset randomly
        dataset_train_ce, dataset_train_mmd, dataset_val, dataset_test = torch.utils.data.random_split(self.dataset, [dataset_size_ce, dataset_size_mmd, dataset_size_val, dataset_size_test])

        #dataloader
        train_ce_loader = DataLoader(dataset=dataset_train_ce,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=2)

        train_mmd_loader = DataLoader(dataset=dataset_train_mmd,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=2)


        val_loader = DataLoader(dataset=dataset_val,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=2)

        test_loader = DataLoader(dataset=dataset_test,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=2)

        data_loader = {}
        data_loader['ce'] = train_ce_loader
        data_loader['mmd'] = train_mmd_loader
        data_loader['val'] = val_loader
        data_loader['test'] = test_loader



        return data_loader