import os, argparse
import torch
import numpy as np
from dataprep import spectralImages_1D
from model import SkipConvNet
import soundfile as sf
from pathlib import Path
from librosa.core import stft, istft
import matplotlib.image as mpimg
import pkbar


def __readscpFiles__(filename):
    with open(filename) as f:
        lines = f.readlines()
    f.close()

    fileList = {}
    for line in lines:
        utt_name, utt_loc, enhance_loc = line.strip().split(' ')
        fileList[utt_name] = {}
        fileList[utt_name]['audioloc'] = utt_loc
        fileList[utt_name]['enhanceloc'] = enhance_loc
    return fileList

def decode(audioName, locations, model, device):
	PSD_frames = spectralImages_1D(audioName, locations['audioloc'])
	nframes = len([key for key in PSD_frames if 'Phase' in key])
	audio = {}
	noisy_mag = []; noisy_phase = []; noisy_norm = [];
	clean_mag = []; clean_phase = []; clean_norm = [];

	for k in range(nframes):
		uttname = 'MagdB_'+audioName+'_frame_'+str(k)
		noisy_mag.append(PSD_frames[uttname])
		noisy_phase.append(PSD_frames[uttname.replace('MagdB', 'Phase')])

	noisy_norm = PSD_frames[uttname.replace('MagdB', 'Norm').split('_frame')[0]]
	samples    = PSD_frames[uttname.replace('MagdB', 'Samples').split('_frame')[0]]
	
	audio['noisy_mag']   = torch.from_numpy(np.expand_dims(noisy_mag, axis=1))
	audio['noisy_phase'] = np.hstack(noisy_phase)
	audio['noisy_norm']  = noisy_norm
	audio['utt_samples'] = int(samples)
	audio['uttname']     = audioName

	with torch.no_grad():
		input_mag     = audio['noisy_mag'].float().to(device)
		enhanced_mag  = model(input_mag).cpu().numpy()
	if enhanced_mag.shape[0]>1:
		enhanced_mag  = np.hstack(np.squeeze(enhanced_mag))
	else:
		enhanced_mag  = np.squeeze(enhanced_mag)
	noisy_mag = np.hstack(np.squeeze(audio['noisy_mag'].numpy()))
	noisy_mag = np.interp(noisy_mag, [-1,1], audio['noisy_norm'])
	enhanced_mag = np.interp(enhanced_mag, [-1,1],audio['noisy_norm'])

	temp = np.zeros((257, enhanced_mag.shape[1])) + 1j*np.zeros((257, enhanced_mag.shape[1]))
	temp[:-1,:] = 10**(enhanced_mag/20) * (np.cos(audio['noisy_phase']) + np.sin(audio['noisy_phase'])*1j)
	enhanced_audio = istft(temp)
	enhanced_audio = 0.98*enhanced_audio/np.max(np.abs(enhanced_audio))
	enhanced_audio = enhanced_audio[:audio['utt_samples']]

	enhanceloc = locations['enhanceloc']
	Path(os.path.dirname(enhanceloc)).mkdir(parents=True, exist_ok=True)
	sf.write(enhanceloc, enhanced_audio,   16000)
	return


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Decode Single Channel Dev & Eval Set - REVERB Challenge')
	parser.add_argument('--audiofilelist', type=str,   help='Location for scp file with audio locations',  required=True)
	parser.add_argument('--specImageDir',  type=str,   help='Location for SpecImages         (default: ./SpecImages)', default=os.getcwd()+'/SpecImages')
	parser.add_argument('--model',         type=str,   help='Absolute path for saved model   (default: ./Saved_Model/best_model.ckpt)', default=os.getcwd()+'/Saved_Model/best_model.ckpt')
	args = parser.parse_args()

	use_cuda = torch.cuda.is_available()
	device    = torch.device('cuda' if use_cuda else 'cpu')
	args.specImageDir = '/data/scratch/vkk160330/Features/Reverb_Spec'    # Comment this for your run
	model = SkipConvNet(args.specImageDir).to(device)
	saved_model = torch.load(args.model)
	model.load_state_dict(saved_model['state_dict'])
	model.eval()

	audiofilelist = __readscpFiles__(args.audiofilelist)
	pbar = pkbar.Pbar(name='Decoding AudioFiles: ', target=len(audiofilelist))
	for i, (audioName, locations) in enumerate(audiofilelist.items()):
		decode(audioName, locations, model, device)	
		pbar.update(i)
