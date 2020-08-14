#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:58:27 2020

@author: vinay
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
from dataloader import SpecImages
import soundfile as sf
from pathlib import Path
from librosa.core import stft, istft
import pkbar
import kaldiio

def __readscpFiles__(filename):
    with open(filename) as f:
        lines = f.readlines()
    f.close()

    fileList = {}
    for line in lines:
        utt_name = line.split(' ')[0].strip()
        utt_loc = line.split(' ')[1].strip()
        fileList[utt_name] = utt_loc
    return fileList


def getfileLocs(reverbscp, wpescp, enhanceName):

	filelocs   = {'uttname':[], 'reverbloc':[], 'wpeloc':[], 'enhanceloc':[]}
	reverblist = __readscpFiles__(reverbscp)
	wpelist    = __readscpFiles__(wpescp)

	for utt in reverblist.keys():
		filelocs['uttname'].append(utt)
		filelocs['reverbloc'].append(reverblist[utt])
		filelocs['wpeloc'].append(wpelist[utt])
		filelocs['enhanceloc'].append(wpelist[utt].replace('WPE', enhanceName))

	filelocs = pd.DataFrame(filelocs)
	return filelocs



datasetloader   = SpecImages('/data/scratch/vkk160330/Features/Reverb_Spec/1ch/Dev', mode='decode')
# def deocdeData(model, device, destfolder):




# def decode(dataset, model, device, destfolder):
# 	kaldidir = '/erasable/dataset/REVERB/kaldi_reverb/vinay_s5/data_se/'
# 	filelocs = getfileLocs(kaldidir+dataset+'/wav.scp', kaldidir+dataset+'_wpe/wav.scp', destfolder)
# 	if 'dt' in dataset:
# 		datasetloader   = ReverbChallenge_Images('/data/scratch/vkk160330/Features/Reverb_Spec/1ch/Dev', mode='decode')
# 	elif 'et' in dataset:
# 		datasetloader   = ReverbChallenge_Images('/data/scratch/vkk160330/Features/Reverb_Spec/1ch/Eval', mode='decode')

# 	with torch.no_grad():
# 		pbar = pkbar.Pbar(name='Decoding : '+dataset, target=len(filelocs))
# 		for i, row in filelocs.iterrows():
# 			audio = datasetloader.__evalaudio__(row['uttname'])
# 			input_mag     = audio['noisy_mag'].unsqueeze(1).to(device)
# 			enhanced_mag  = model(input_mag).cpu().numpy()
# 			if enhanced_mag.shape[0]>1:
# 				enhanced_mag  = np.hstack(np.squeeze(enhanced_mag))
# 			else:
# 				enhanced_mag  = np.squeeze(enhanced_mag)
# 			enhanced_mag  = np.interp(enhanced_mag, [-1,1],audio['noisy_norm'])
# 			temp = np.zeros((257, enhanced_mag.shape[1])) + 1j*np.zeros((257, enhanced_mag.shape[1]))
# 			temp[:-1,:] = 10**(enhanced_mag/20) * (np.cos(audio['noisy_phase']) + np.sin(audio['noisy_phase'])*1j)
# 			enhanced_audio = istft(temp)
# 			enhanced_audio = 0.98*enhanced_audio/np.max(np.abs(enhanced_audio))
# 			enhanced_audio = enhanced_audio[:audio['utt_samples']]
# 			Path(os.path.dirname(row['enhanceloc'])).mkdir(parents=True, exist_ok=True)
# 			sf.write(row['enhanceloc'], enhanced_audio,   16000)
# 			del audio,input_mag,enhanced_mag,temp,enhanced_audio
# 			pbar.update(i)
# 	return



# def main():
# 	use_cuda  = torch.cuda.is_available()
# 	device    = torch.device('cuda' if use_cuda else 'cpu')
# 	dtype     = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# 	model = SkipConvNet().to(device)
# 	saved_model = torch.load('./lightning_logs/version_0/checkpoints/epoch=21.ckpt')
# 	model.load_state_dict(saved_model['state_dict'])
# 	model.eval()
# 	destfolder= 'SkipConvNet'
# 	print('Decoding audiofiles')
# 	for dataset in ['dt_real_1ch', 'et_real_1ch', 'dt_simu_1ch', 'et_simu_1ch']:
# 		decode(dataset, model, device,destfolder)


# if __name__ == "__main__":
#     main()
#     print('Done!')

