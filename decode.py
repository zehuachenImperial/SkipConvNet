#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:58:27 2020

@author: vinay
"""
import os, argparse
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
	parser = argparse.ArgumentParser(description='Decode Single Channel Dev & Eval Set - REVERB Challenge')
	parser.add_argument('--dataloc',      type=str,   help='Location for scp Files       (default: ./Data)', default='./Data')
	parser.add_argument('--specImageloc', type=str,   help='Location for SpecImages      (default: ./SpecImages)', default='./SpecImages')
	parser.add_argument('--destfolder',   type=str,   help='Location for enhanced audiofiles        (default: ./Enhanced)', default='./Enhanced')
	parser.add_argument('--model',        type=str,   help='Absolute path for saved model           (default: ./chkpts/best.ckpt)', default='./chkpts/best.ckpt')
	args = parser.parse_args()
	
	
	device    = torch.device('cuda' if use_cuda else 'cpu')
	model = SkipConvNet().to(device)
	saved_model = torch.load(args.model)
	model.load_state_dict(saved_model['state_dict'])
	model.eval()
	deocdeData(args.dataloc, args.specImageloc, args.destfolder, model, device)

