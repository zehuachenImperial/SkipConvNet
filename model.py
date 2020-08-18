import torch
import torch.nn as nn
from torch.nn import functional as F
from dataloader import SpecImages
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary

class Conv2d(nn.Module):
    """ 
    Convolutional Module with weights initialized with normal distribution and weights to zeros
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2, stride=2):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                              padding=padding, stride=2, bias=True)
        torch.nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.conv.bias)
    def forward(self, x):
        return self.conv(x)


class ConvTranspose2d(nn.Module):
    """ 
    Transpose Convolution Module with weights initialized with normal distribution and weights to zeros
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(ConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, bias=True)
        torch.nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.conv.bias)
    def forward(self, x):
        return self.conv(x)

class SkipBlock(nn.Module):
    """ 
    Each SkipBlock is a Activation -> Convolutions + Residual Connection followed by a normalization 
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super(SkipBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                              padding=padding, bias=True)
        torch.nn.init.normal_(self.conv1.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.conv1.bias)
        self.norm    = nn.BatchNorm2d(in_channels)
        self.lRelu   = nn.LeakyReLU(negative_slope=0.2)
        
    def forward(self, x):
        return self.norm(self.conv1(self.lRelu(x)) + self.lRelu(x))

class SkipConnection(nn.Module):
    """ 
    SkipConnection is a concatenations of SkipBlocks
    """
    def __init__(self, in_channels, num_convblocks):
        super(SkipConnection,self).__init__()
        self.skip_blocks = [SkipBlock(in_channels, in_channels, kernel_size=3, padding=1) for k in range(num_convblocks)]
        self.skip_path   = nn.Sequential(*self.skip_blocks)
        
    def forward(self, x):
        return self.skip_path(x)

class SkipConvNet(pl.LightningModule): 
    """ 
    Proposed: SkipConvNet (Interspeech 2020)
    """
    def __init__(self, SpecImageDir):
        super(SkipConvNet, self).__init__()
        self.modelName    = 'SkipConvNet'
        self.SpecImageDir = SpecImageDir

        self.dconv1 = Conv2d(in_channels=1,   out_channels=64,  kernel_size=5, padding=2)
        self.skip1  = SkipConnection(in_channels=64, num_convblocks=8)

        self.dconv2 = Conv2d(in_channels=64,  out_channels=128,  kernel_size=5, padding=2)
        self.dBNorm2   = nn.BatchNorm2d(128)
        self.skip2  = SkipConnection(in_channels=128, num_convblocks=8)

        self.dconv3 = Conv2d(in_channels=128, out_channels=256,  kernel_size=5, padding=2)
        self.dBNorm3   = nn.BatchNorm2d(256)
        self.skip3  = SkipConnection(in_channels=256, num_convblocks=4)

        self.dconv4 = Conv2d(in_channels=256, out_channels=512,  kernel_size=5, padding=2)
        self.dBNorm4   = nn.BatchNorm2d(512)
        self.skip4  = SkipConnection(in_channels=512, num_convblocks=4)

        self.dconv5 = Conv2d(in_channels=512, out_channels=512,  kernel_size=5, padding=2)
        self.dBNorm5   = nn.BatchNorm2d(512)
        self.skip5  = SkipConnection(in_channels=512, num_convblocks=2)

        self.dconv6 = Conv2d(in_channels=512, out_channels=512,  kernel_size=5, padding=2)
        self.dBNorm6   = nn.BatchNorm2d(512)
        self.skip6  = SkipConnection(in_channels=512, num_convblocks=2)

        self.dconv7 = Conv2d(in_channels=512, out_channels=512,  kernel_size=5, padding=2)
        self.dBNorm7   = nn.BatchNorm2d(512)
        self.skip7  = SkipConnection(in_channels=512, num_convblocks=1)

        self.dconv8 = Conv2d(in_channels=512, out_channels=512,  kernel_size=5, padding=2)
        
        self.uconv1  = nn.ConvTranspose2d(in_channels=512,  out_channels=512, kernel_size=2, stride=2)
        self.uBNorm1 = nn.BatchNorm2d(512)
        
        self.uconv2  = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.uBNorm2 = nn.BatchNorm2d(512)
        
        self.uconv3  = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.uBNorm3 = nn.BatchNorm2d(512)
        
        self.uconv4  = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.uBNorm4 = nn.BatchNorm2d(512)
        
        self.uconv5  = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=2, stride=2)
        self.uBNorm5 = nn.BatchNorm2d(256)
        
        self.uconv6  = nn.ConvTranspose2d(in_channels=512, out_channels=128,  kernel_size=2, stride=2)
        self.uBNorm6 = nn.BatchNorm2d(128)
        
        self.uconv7  = nn.ConvTranspose2d(in_channels=256, out_channels=64,   kernel_size=2, stride=2)
        self.uBNorm7 = nn.BatchNorm2d(64)
        
        self.uconv8  = nn.ConvTranspose2d(in_channels=128, out_channels=1,    kernel_size=2, stride=2)
        
        self.lRelu   = nn.LeakyReLU(negative_slope=0.2)
        self.relu    = nn.ReLU()
        self.tanh    = nn.Tanh()
        self.drop    = nn.Dropout(0.5)
        
    
    def forward(self, x):
        
        # +++++++++++++++++++ Squeezing Path  +++++++++++++++++++++ #
        d1 = self.dconv1(x)
        d2 = self.dBNorm2(self.dconv2(self.lRelu(d1)))
        d3 = self.dBNorm3(self.dconv3(self.lRelu(d2)))
        d4 = self.dBNorm4(self.dconv4(self.lRelu(d3)))
        d5 = self.dBNorm5(self.dconv5(self.lRelu(d4)))
        d6 = self.dBNorm6(self.dconv6(self.lRelu(d5)))
        d7 = self.dBNorm7(self.dconv7(self.lRelu(d6)))
        d8 = self.dconv8(self.lRelu(d7))
        
        # +++++++++++++++++++ Expanding Path  +++++++++++++++++++++ #
        u1 = self.drop(self.uBNorm1(self.uconv1(self.relu(d8))))
        u2 = self.drop(self.uBNorm2(self.uconv2(self.relu(torch.cat((u1, self.skip7(d7)), 1)))))
        u3 = self.drop(self.uBNorm3(self.uconv3(self.relu(torch.cat((u2, self.skip6(d6)), 1))))) 
        u4 = self.uBNorm4(self.uconv4(self.relu(torch.cat((u3, self.skip5(d5)), 1)))) 
        u5 = self.uBNorm5(self.uconv5(self.relu(torch.cat((u4, self.skip4(d4)), 1)))) 
        u6 = self.uBNorm6(self.uconv6(self.relu(torch.cat((u5, self.skip3(d3)), 1)))) 
        u7 = self.uBNorm7(self.uconv7(self.relu(torch.cat((u6, self.skip2(d2)), 1)))) 
        u8 = self.uconv8(self.relu(torch.cat((u7, self.skip1(d1)), 1)))
        Output = self.tanh(u8)
        return Output  

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        return {'val_loss': F.mse_loss(y_hat, y)}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        return {'test_loss': F.mse_loss(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': logs} #, 'progress_bar': logs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-5, betas=(0.9,0.999))
        scheduler = ReduceLROnPlateau(optimizer, mode='min',  patience=1, verbose=True)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        TrainData   = SpecImages(self.SpecImageDir+'/1ch/Train', mode='train')
        trainloader = DataLoader(TrainData, batch_size=4, shuffle=True,  num_workers=8)
        return trainloader

    def val_dataloader(self):
        DevData   = SpecImages(self.SpecImageDir+'/1ch/Dev', mode='train')
        devloader = DataLoader(DevData, batch_size=4, shuffle=False,  num_workers=8)
        return devloader

    def test_dataloader(self):
        EvalData   = SpecImages(self.SpecImageDir+'/1ch/Eval', mode='train')
        evalloader = DataLoader(EvalData, batch_size=4, shuffle=False,  num_workers=8)
        return evalloader
        

if __name__=='__main__':
    SpecImageDir = '/data/scratch/vkk160330/Features/Reverb_Spec'
    model = SkipConvNet(SpecImageDir).to('cuda')
    summary(model, input_size=(1,256,256), batch_size=1, device='cuda')