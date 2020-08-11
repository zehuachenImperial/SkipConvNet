import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import kaldiio


class SpecImages(Dataset):
    def __init__(self,datadir, mode='train'):
        self.reverbFiles = self.__readscpFiles__(self.local+datadir+'/Sim/wav.scp')
        if mode == 'train':
            self.cleanFiles  = self.__readscpFiles__(self.local+datadir+'/Clean/wav.scp')
        if 'Train' not in datadir:
            self.realFiles   = self.__readscpFiles__(self.local+datadir+'/Real/wav.scp')
            self.allFiles    = {}
            self.allFiles.update(self.reverbFiles)
            self.allFiles.update(self.realFiles)
            self.alluttID    = list(self.allFiles.keys())
            self.alluttID    = [n for n in self.alluttID if 'MagdB' in n]
            self.allaudioID  = list(set([n.split('_frame')[0] for n in self.alluttID]))

        self.uttID    = list(self.reverbFiles.keys())
        self.uttID    = [n for n in self.uttID if 'MagdB' in n]
        self.audioID  = list(set([n.split('_frame')[0] for n in self.uttID]))


    def __len__(self):
        return len(self.uttID)

    def __audiolen__(self):
        return len(self.audioID)

    def __getitem__(self, idx):
        batch = {}
        uttname = self.uttID[idx]

        batch['noisy_mag'] = np.expand_dims(kaldiio.load_mat(self.local+self.reverbFiles[uttname]), axis=0)
        batch['clean_mag'] = np.expand_dims(kaldiio.load_mat(self.local+self.cleanFiles[uttname]), axis=0)

        batch['noisy_mag'] = torch.from_numpy(np.float32(batch['noisy_mag']))
        batch['clean_mag'] = torch.from_numpy(np.float32(batch['clean_mag']))
        return batch['noisy_mag'], batch['clean_mag']

    def __readscpFiles__(self, filename):
        with open(filename) as f:
            lines = f.readlines()
        f.close()

        fileList = {}
        for line in lines:
            utt_name = line.split(' ')[0]
            utt_loc = line.split(' ')[1]
            fileList[utt_name] = utt_loc
        return fileList


    def __getsample__(self, idx):
        batch = {}
        uttname = self.uttID[idx]

        batch['noisy_mag'] = np.expand_dims(kaldiio.load_mat(self.local+self.reverbFiles[uttname]), axis=0)
        batch['clean_mag'] = np.expand_dims(kaldiio.load_mat(self.local+self.cleanFiles[uttname]), axis=0)

        uttname = uttname.replace('MagdB', 'Phase')
        batch['noisy_phase'] = kaldiio.load_mat(self.local+self.reverbFiles[uttname])
        batch['clean_phase'] = kaldiio.load_mat(self.local+self.cleanFiles[uttname])


        uttname = uttname.replace('Phase', 'Norm').split('_frame')[0]
        batch['noisy_norm'] = kaldiio.load_mat(self.local+self.reverbFiles[uttname])
        batch['clean_norm'] = kaldiio.load_mat(self.local+self.cleanFiles[uttname])


        batch['noisy_mag'] = torch.from_numpy(np.float32(batch['noisy_mag']))
        batch['clean_mag'] = torch.from_numpy(np.float32(batch['clean_mag']))

        uttname = uttname.replace('Norm', 'Samples')
        batch['samples']   =  kaldiio.load_mat(self.local+self.reverbFiles[uttname.split('_frame')[0]])
        return batch


    def __getevalsample__(self, idx):
        batch = {}
        uttname = self.alluttID[idx]

        batch['noisy_mag'] = np.expand_dims(kaldiio.load_mat(self.local+self.allFiles[uttname]), axis=0)

        uttname = uttname.replace('MagdB', 'Phase')
        batch['noisy_phase'] = kaldiio.load_mat(self.local+self.allFiles[uttname])

        uttname = uttname.replace('Phase', 'Norm').split('_frame')[0]
        batch['noisy_norm'] = kaldiio.load_mat(self.local+self.allFiles[uttname])

        batch['noisy_mag'] = torch.from_numpy(np.float32(batch['noisy_mag']))

        uttname = uttname.replace('Norm', 'Samples')
        batch['samples']   =  kaldiio.load_mat(self.local+self.allFiles[uttname.split('_frame')[0]])
        return batch


    def __getaudio__(self, idx):
        uttname = self.uttID[idx].split('_frame_')[0]
        idx = self.audioID.index(uttname)
        frame_idx = [n for n in range(len(self.uttID)) if self.audioID[idx] in self.uttID[n]]

        audio = {}
        noisy_mag = []; noisy_phase = []; noisy_norm = [];
        clean_mag = []; clean_phase = []; clean_norm = [];
        for k in range(len(frame_idx)):
            sample = self.__getsample__(frame_idx[k])
            noisy_mag.append(sample['noisy_mag'])
            clean_mag.append(sample['clean_mag'])
            noisy_phase.append(sample['noisy_phase'])
            clean_phase.append(sample['clean_phase'])
            noisy_norm.append(sample['noisy_norm'])
            clean_norm.append(sample['clean_norm'])


        audio['noisy_mag']   = torch.cat(noisy_mag, dim=0)
        audio['clean_mag']   = torch.cat(clean_mag, dim=0)
        audio['noisy_phase'] = np.hstack(noisy_phase)
        audio['clean_phase'] = np.hstack(clean_phase)
        audio['noisy_norm']  = sample['noisy_norm']
        audio['clean_norm']  = sample['clean_norm']
        audio['utt_samples']  = int(sample['samples'][0])
        audio['uttname']     = self.audioID[idx]
        return audio


    def __getsingleaudio__(self, scpfile):
        #idx = self.allaudioID.index('MagdB_'+uttname)
        #uttname = self.alluttID[idx]
        audiofile = self.__readscpFiles__(scpfile)
        audio = {}; batch = {};
        noisy_mag = []; noisy_phase = []; noisy_norm = [];
        for k in audiofile.keys():
            if 'MagdB' in k:
                uttname_original = audiofile[k].strip()

                uttname_original = k
                batch['noisy_mag'] = np.expand_dims(kaldiio.load_mat(audiofile[k]), axis=0)
                batch['noisy_mag'] = torch.from_numpy(np.float32(batch['noisy_mag']))

                uttname = uttname_original.replace('MagdB', 'Phase')
                batch['noisy_phase'] = kaldiio.load_mat(audiofile[uttname])

                uttname = uttname.replace('Phase', 'Norm')
                batch['noisy_norm'] = kaldiio.load_mat(audiofile[uttname.split('_frame')[0]])

                uttname = uttname.replace('Norm', 'Samples')
                batch['samples']   =  kaldiio.load_mat(audiofile[uttname.split('_frame')[0]])

                noisy_mag.append(batch['noisy_mag'])
                noisy_phase.append(batch['noisy_phase'])
                noisy_norm.append(batch['noisy_norm'])

        audio['noisy_mag']   = torch.cat(noisy_mag, dim=0)
        audio['noisy_phase'] = np.hstack(noisy_phase)
        audio['noisy_norm']  = batch['noisy_norm']
        audio['utt_samples'] = batch['samples'][0]
        return audio

    def __evalaudio__(self, uttname):
        idx = self.allaudioID.index('MagdB_'+uttname)
        frame_idx = [n for n in range(len(self.alluttID)) if self.allaudioID[idx] in self.alluttID[n]]

        audio = {}
        noisy_mag = []; noisy_phase = []; noisy_norm = [];
        for k in range(len(frame_idx)):
            sample = self.__getevalsample__(frame_idx[k])
            noisy_mag.append(sample['noisy_mag'])
            noisy_phase.append(sample['noisy_phase'])
            noisy_norm.append(sample['noisy_norm'])


        audio['noisy_mag']   = torch.cat(noisy_mag, dim=0)
        audio['noisy_phase'] = np.hstack(noisy_phase)
        audio['noisy_norm']  = sample['noisy_norm']
        audio['utt_samples'] = int(sample['samples'][0])
        audio['uttname']     = self.allaudioID[idx]
        return audio
