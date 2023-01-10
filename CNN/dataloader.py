import re
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
import os
from torch.utils.data import Dataset
from copy import deepcopy

class SpeechDataset(Dataset):

    def __init__(self, flac_dir, load_method):
        self.audio_path_list = sorted(self.find_files(flac_dir))  # do we need them to be sorted?
        methods = {"librosa": self.librosa_flac2melspec, "soundfile": self.sf_loader, "torchaudio": self.torch_flac2melspec}
        self.labels = {"es": 0, "en": 1, "de": 2}
        self.languages = {v: k for k, v in self.labels.items()}
        self.chosen_method = methods[load_method]
        
        
    def __len__(self):
        return len(self.audio_path_list)

    def __getitem__(self, index):
        audio_file = self.audio_path_list[index]  
        label = self.labels[self.get_label(audio_file)]
        spectro, _ = self.chosen_method(audio_file)  # _ is fs
        #spectro = torch.from_numpy(spectro)
        #spectro = spectro.unsqueeze(0)
        return spectro, label

    def find_files(self, directory, pattern=".flac"):
        """
        Recursive search method to find files. Credit to Paul Magron and Andrea de Marco
        for OG method
        """

        return  [f.path for f in os.scandir(directory) if f.path.endswith(pattern)]  # ends with does not like regex
    
    def monolingual_path_list(self,language):
        """
        Input: string ('en'|'de'|'es')
        Output: audio path list only containing file names of the chosen language
        """
        r = re.compile(rf'.*\/{language}_.*\.flac') 
        newlist = list(filter(r.match, self.audio_path_list))
        return newlist
    
    def get_label(self, path):
        labelRegex = re.compile(r"(es|en|de)_.*.flac")
        patterns = labelRegex.findall(path)
        assert len(patterns) == 1
        return patterns[0]

    def torch_flac2melspec(self, file_path):
        waveform, sample_rate = torchaudio.load(file_path, normalize=True)
        transform = T.MelSpectrogram(sample_rate)        
        return transform(waveform), sample_rate
        
    def sf_loader(self, file_path):
        with open(file_path, "rb") as f:
            data, samplerate = sf.read(f)
        return data, samplerate

    def librosa_flac2melspec(self, file_path, n_mels=64, melspec_size=512, visual=False):
        """
        the librosa method we are using atm
        """
        sig, fs =  librosa.load(file_path, sr=None)
        sig /= np.max(np.abs(sig), axis=0)
        n_fft = melspec_size
        hop_length = int(n_fft/2)

        # padding signal if less than a second
        if len(sig) < fs:
            padded_array = np.zeros(fs)
            padded_array[:np.shape(sig)[0]] = sig
            sig = padded_array

        melspec = librosa.feature.melspectrogram(y=sig, sr=fs,
                                                 center=True, n_fft=n_fft,
                                                 hop_length=hop_length, n_mels=n_mels)


        if visual:
            self.plotmelspec(melspec, fs, hop_length)

        melspec = librosa.power_to_db(melspec, ref=1.0)
        melspec /= 80.0  # highest db...
        melspec = self.checkmelspec(melspec)
        return melspec, fs

    def checkmelspec(self, melspec, n_mels=64):
        """
        this method works with librosa

        """
        if melspec.shape[1] < n_mels:  # n_mels
            shape = np.shape(melspec)
            padded_array = np.zeros((shape[0], n_mels)) - 1
            padded_array[0:shape[0], :shape[1]] = melspec
            melspec = padded_array
        return melspec

    def plotmelspec(self, melspec, fs, hop_length, show=False):
        plt.figure(figsize=(8, 6))
        plt.xlabel("Time")
        plt.ylabel("Mel-Frequency")
        librosa.display.specshow(librosa.power_to_db(melspec, ref=np.max),
                                 y_axis="mel", fmax=fs/2, sr=fs,
                                 hop_length=hop_length, x_axis="time")
        plt.colorbar(format="%+2.0f db")
        plt.title("Mel Spectogram")
        plt.tight_layout()
        if show:
            plt.show()
        plt.close()  # to close windows and fix warning!



def get_balanced_subset(train_data,n):
    '''
    Creates subset with equal amounts of data from each language
    Input: instance of class Speechdataset, desired number of files per language (int) 
    Output: balanced subset (torch.utils.data.dataset.ConcatDataset)
    '''
    #create new instances for each language
    en=deepcopy(train_data)
    es=deepcopy(train_data)
    de=deepcopy(train_data)
    
    #modify the audio_path_list to only include paths to files of a single language
    en.audio_path_list=en.monolingual_path_list('en')
    es.audio_path_list=es.monolingual_path_list('es')
    de.audio_path_list=de.monolingual_path_list('de')
    
    #extract equally sized subsets from each monolingual dataset
    en_sub = Subset(en, torch.arange(n))
    es_sub = Subset(es, torch.arange(n))
    de_sub = Subset(de, torch.arange(n))
    
    subset=en_sub+de_sub+es_sub #concatenate subsets
    return subset