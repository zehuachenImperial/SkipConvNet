#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:58:27 2020

@author: vinay
"""
import torch
import torch.nn as nn
import numpy as np
import os, argparse
import soundfile as sf
from pathlib import Path
from dataloader import SpecImages
from librosa.core import stft, istft
import pkbar, kaldiio
from model import SkipConvNet

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


def deocdeData(dataloc, specImageloc, destfolder, model, device): 
	for dataset in ['Dev', 'Eval']:
		audiofiles  = __readscpFiles__(dataloc+'/'+dataset+'_SimData.scp')
		audiofiles.update(__readscpFiles__(dataloc+'/'+dataset+'_RealData.scp'))
		pbar = pkbar.Pbar(name='Decoding '+dataset+' AudioFiles: ', target=len(audiofiles))
		
		data = SpecImages(specImageloc+'/'+dataset, mode='decode')
		with torch.no_grad():
			for i, (k, v) in enumerate(audiofiles.items()):
				uttID = data.uttname2idx('MagdB_'+k)
				audio = data.__getaudio__(uttID)
				input_mag     = audio['noisy_mag'].unsqueeze(1).to(device)
				enhanced_mag  = model(input_mag).cpu().numpy()
				if enhanced_mag.shape[0]>1:
					enhanced_mag  = np.hstack(np.squeeze(enhanced_mag))
				else:
					enhanced_mag  = np.squeeze(enhanced_mag)
				enhanced_mag  = np.interp(enhanced_mag, [-1,1],audio['noisy_norm'])
				temp = np.zeros((257, enhanced_mag.shape[1])) + 1j*np.zeros((257, enhanced_mag.shape[1]))
				temp[:-1,:] = 10**(enhanced_mag/20) * (np.cos(audio['noisy_phase']) + np.sin(audio['noisy_phase'])*1j)
				enhanced_audio = istft(temp)
				enhanced_audio = 0.98*enhanced_audio/np.max(np.abs(enhanced_audio))
				enhanced_audio = enhanced_audio[:audio['utt_samples']]

				destloc = destfolder+v.split('Reverb_Challenge')[1]
				Path(os.path.dirname(destloc)).mkdir(parents=True, exist_ok=True)
				sf.write(destloc, enhanced_audio,   16000)
				del audio,input_mag,enhanced_mag,temp,enhanced_audio
				pbar.update(i)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Demo Script to Enhance a list of Reverberant audio files')
	parser.add_argument('--dataloc',      type=str,   help='Location for scp Files       (default: ./Data)', default=os.getcwd()+'/Data')
	parser.add_argument('--specImageDir', type=str,   help='Location for SpecImages      (default: ./SpecImages)', default=os.getcwd()+'/SpecImages')
	parser.add_argument('--destfolder',   type=str,   help='Location for enhanced audiofiles        (default: ./Enhanced)', default=os.getcwd()+'/Enhanced')
	parser.add_argument('--model',        type=str,   help='Absolute path for saved model           (default: ./Saved_Model/best_model.ckpt)', default=os.getcwd()+'/Saved_Model/best_model.ckpt')
	args = parser.parse_args()
	
	args.specImageDir = '/data/scratch/vkk160330/Features/Reverb_Spec'    # Comment this for your run
	device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = SkipConvNet(args.specImageDir).to(device)
	saved_model = torch.load(args.model)
	model.load_state_dict(saved_model['state_dict'])
	model.eval()
	deocdeData(args.dataloc, args.specImageDir, args.destfolder, model, device)

