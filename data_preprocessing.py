import torch
import fnmatch
import matplotlib.pyplot as plt
import soundfile as sf
import os
from torch.utils.data import DataLoader, Dataset, random_split, Subset


# used to clean up es files
# ls /media/andres/2D2DA2454B8413B5/test/test/ | grep -o '.....$' | uniq   # exploring file types
#/media/andres/2D2DA2454B8413B5/test/test/ | grep -o '^es.*'
base_path = "/media/andres/2D2DA2454B8413B5/"
train_dir = base_path + "train/train/"
test_dir = base_path + "test/test/"

def flac_to_spectro(file_path):
    waveform, sample_rate = torchaudio.load(file_path, normalize=True)
    transform = T.MelSpectrogram(sample_rate)
    mel_specgram = transform(waveform)
    return mel_specgram  

class SpeechDataset(Dataset):

    def __init__(self, flac_dir):
        self.audio_path_list = sorted(self.find_files(flac_dir))
        
        # self.specto_data = []
        # for f_path in self.audio_path_list:
        #     self.specto_data.append(flac_to_spectro(f_path))
        

    def __len__(self):
        return len(self.audio_path_list)

    def __getitem__(self, index):
        audio_path = self.audio_path_list[index]
        with open(audio_path, "rb") as f:
            data, samplerate = sf.read(f)
        return data, samplerate
        #return self.specto_data[index]  if using mathilde's method

    def find_files(self, directory, pattern="*.flac"):
        """
        Recursive search method to find files. Credit to Paul Magron for OG method
        """
        audio_path_list = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                audio_path_list.append(os.path.join(root, filename))

        return audio_path_list

test_data = SpeechDataset(test_dir)  # 540
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

for data, samplerate in test_dataloader:
    print(data, samplerate)
    print()
    print(data.shape, samplerate.shape)
    break

raise SystemExit
train_data = Subset(train_data, torch.arange(240)) # 80% 
test_data = Subset(test_data, torch.arange(60))  # 20%

train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)
