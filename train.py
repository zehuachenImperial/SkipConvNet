import os
from model import SkipConvNet
import pytorch_lightning as pl

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"
skipconv_model = SkipConvNet()
trainer = pl.Trainer(max_epochs=40, gpus=3, distributed_backend='ddp')    
trainer.fit(skipconv_model)   
trainer.test()
