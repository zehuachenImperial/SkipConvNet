import os, argparse
from model import SkipConvNet
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train SkipConvNet Model for Single Channel Speech Dereverberation (Interspeech 2020)')
	parser.add_argument('--specImageDir',   type=str,   help='Path to directory with SpecImages       (default: ./SpecImages)', default=os.getcwd()+'/SpecImages')
	parser.add_argument('--epocs',          type=int,   help='Max number of epocs       (default: 15)', default=40)
	parser.add_argument('--gpuIDs',         type=str,   help='GPU to use (list of IDs)  (default: 5,6,7)', default='5,6,7')
	parser.add_argument('--dryrun',         type=str,   help='Dry run for on one batch  (default: False)', default='False')
	args = parser.parse_args()
	
	args.specImageDir = '/data/scratch/vkk160330/Features/Reverb_Spec'    # Comment this for your run
	skipconv_model = SkipConvNet(args.specImageDir)
	if args.dryrun == True:
		trainer = pl.Trainer(fast_dev_run=True)
		trainer.fit(skipconv_model)   
	else:
		gpuIDs  = [int(gpu_id) for gpu_id in args.gpuIDs.split(',')]
		trainer = pl.Trainer(max_epochs=args.epocs, gpus=gpuIDs, distributed_backend='ddp', precision=16)  
		trainer.fit(skipconv_model)   
		trainer.test()
