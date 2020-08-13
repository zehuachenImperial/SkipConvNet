import os, argparse
from model import SkipConvNet
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train SkipConvNet Model for Single Channel Speech Dereverberation (Interspeech 2020)')
	parser.add_argument('--epocs',    type=int,   help='Max number of epocs       (default: 15)', default=15)
	parser.add_argument('--ngpus',    type=int,   help='Max number of GPUs to use (default:  3)', default=3)
	parser.add_argument('--precision',type=int,   help='Precision for Loss        (default: 16)', default=16)
	parser.add_argument('--logdir',   type=str,   help='Tensorflow logdir name    (default: tfLogs)', default='tfLogs')
	parser.add_argument('--chkptdir', type=str,   help='Checkpoints dir name      (default: chkpts)', default='chkpts')
	args = parser.parse_args()
	
	skipconv_model = SkipConvNet()
	checkpoint_callback = ModelCheckpoint(filepath=os.getcwd()+'/'+args.chkptdir, save_top_k=True, verbose=True, monitor='val_loss',mode='min',prefix='')

	logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name=args.logdir)
	trainer = pl.Trainer(max_epochs=40, gpus=3, distributed_backend='ddp',logger=logger, precision=args.precision, checkpoint_callback=ModelCheckpoint())  
	trainer.fit(skipconv_model)   
	trainer.test()
