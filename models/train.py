from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
from torchvision.models import resnet18
from typing import Optional

class myLightningModule(LightningModule):
    '''
    This training code follows the standard structure of Pytorch Lightning. It's worth looking at their docs for a more in-depth dive as to why it is structured this way.
    '''
    
    def __init__(self,
                learning_rate=1e-3,
                total_steps: int = 200000,
                train_batch_size: int = 64,              
                **kwargs,
                ):

        super().__init__()
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss()
        # Define the ResNet backbone
        self.backbone = resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features

        # Replace the fully connected layer with a custom classification head
        self.backbone.fc = nn.Linear(num_features, 4)  # 4 classes for multi-class classification

    def forward(self, input):
        # Forward pass through the ResNet model
        return self.backbone(input)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # Separate the batch into input and target
        input, target = batch[0], batch[1]
        out = self.forward(input)
        loss = self.loss(out, target)
        # Log the training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
      
    def validation_step(self, batch, batch_idx):
        # Separate the batch into input and target
        input, target = batch[0], batch[1]
        out = self.forward(input)
        loss = self.loss(out, target)
        # Log the validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, eps=1e-8)
        # Define scheduler here if needed
        return [optimizer]
