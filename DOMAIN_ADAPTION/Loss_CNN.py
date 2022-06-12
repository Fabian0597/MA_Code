import torch

class Loss_CNN():
    def __init__(self, model_cnn, model_fc, criterion, MMD_loss_calculator, MMD_loss_flag_phase, GAMMA):
        self.model_cnn = model_cnn
        self.model_fc = model_fc
        self.criterion = criterion
        self.MMD_loss_calculator = MMD_loss_calculator
        self.MMD_loss_flag_phase = MMD_loss_flag_phase
        self.GAMMA = GAMMA

    
    def forward(self, batch_data, labels_source, labels_target):
        #Feature extraction
        x_conv_1, x_conv_2, x_conv_3, x_flatten, x_fc1 = self.model_cnn(batch_data.float())
        x_fc2, x_fc3 = self.model_fc(x_fc1)
        
        batch_size = len(labels_source)   

        #CE Loss
        source_ce_loss = self.criterion(x_fc3[:batch_size, :], labels_source)
        target_ce_loss = self.criterion(x_fc3[batch_size:, :], labels_target)
        
        
        #MMD Loss for FC Layers
        #mmd_loss_1_fc = self.MMD_loss_calculator.forward(x_flatten[:batch_size, :], x_flatten[batch_size:, :])
        #mmd_loss_2_fc = self.MMD_loss_calculator.forward(x_fc1[:batch_size, :], x_fc1[batch_size:, :])
        #mmd_loss_3_fc = self.MMD_loss_calculator.forward(x_fc2[:batch_size, :], x_fc2[batch_size:, :])
        

        #MMD Loss for CNN Layers
        mmd_loss_1_cnn = 0
        mmd_loss_2_cnn = 0
        mmd_loss_3_cnn = 0
        
        for channel1 in range(x_conv_1.size()[1]):
            mmd_loss_1_cnn += self.MMD_loss_calculator.forward(x_conv_1[:batch_size, channel1, :], x_conv_1[batch_size:,channel1, :])
        for channel2 in range(x_conv_2.size()[1]):
            mmd_loss_2_cnn += self.MMD_loss_calculator.forward(x_conv_2[:batch_size, channel2, :], x_conv_2[batch_size:,channel2, :])
        for channel3 in range(x_conv_3.size()[1]):
            mmd_loss_3_cnn += self.MMD_loss_calculator.forward(x_conv_3[:batch_size, channel3, :], x_conv_3[batch_size:,channel3, :])
          
        
        #Total MMD Loss
        mmd_loss =  mmd_loss_1_cnn + mmd_loss_2_cnn + mmd_loss_3_cnn

        # list of latent space features in FC1 for plot
        class_0_source_fc2 = x_fc2[:batch_size, :][labels_source==0]
        class_1_source_fc2 = x_fc2[:batch_size, :][labels_source==1]
        class_0_target_fc2 = x_fc2[batch_size:, :][labels_target==0]
        class_1_target_fc2 = x_fc2[batch_size:, :][labels_target==1]
        
        #Accuracy Source
        argmax_source_pred = torch.argmax(x_fc3[:batch_size, :], dim=1)
        result_source_pred = argmax_source_pred == labels_source
        correct_source_pred = result_source_pred[result_source_pred == True]
        acc_total_source = 100 * len(correct_source_pred)/len(labels_source)
        
        #Accuracy Target
        argmax_target_pred = torch.argmax(x_fc3[batch_size:, :], dim=1)
        result_target_pred = argmax_target_pred == labels_target
        correct_target_pred = result_target_pred[result_target_pred == True]
        acc_total_target = 100 * len(correct_target_pred)/len(labels_target)
        

        # Separation between MMD and CE Train Phase
        if self.MMD_loss_flag_phase == True:
            loss = source_ce_loss + mmd_loss
        else:
            loss = source_ce_loss

        
        return loss, mmd_loss, source_ce_loss, target_ce_loss, acc_total_source, acc_total_target, class_0_source_fc2, class_1_source_fc2, class_0_target_fc2, class_1_target_fc2
    
