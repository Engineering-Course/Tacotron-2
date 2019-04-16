import numpy as np
from datasets.audio import *
import os
from hparams import hparams

# n_sample = 0 #Change n_steps here
# mel_folder = 'logs-Tacotron/mel-spectrograms' #Or change file path
# mel_file = 'mel-prediction-step-{}.npy'.format(n_sample) #Or file name (for other generated mels)
out_dir = 'wav_out'

os.makedirs(out_dir, exist_ok=True)

#mel_file = os.path.join(mel_folder, mel_file)
mel_file = 'data/training_data/mels/mel-LJ001-0001.npy'
mel_spectro = np.load(mel_file)
print (mel_spectro.shape)

wav = inv_mel_spectrogram(mel_spectro.T, hparams)
#save the wav under test_<folder>_<file>
save_wav(wav, os.path.join(out_dir, 'test_mel_{}.wav'.format(mel_file.replace('/', '_').replace('\\', '_').replace('.npy', ''))),
        sr=hparams.sample_rate)


from tacotron.utils.plot import *

plot_spectrogram(mel_spectro, path=os.path.join(out_dir, 'test_mel_{}.png'.format(mel_file.replace('/', '_').replace('\\', '_').replace('.npy', ''))))


lin_file = 'data/training_data/linear/linear-LJ001-0001.npy'
lin_spectro = np.load(lin_file)
print (lin_spectro.shape)


wav = inv_linear_spectrogram(lin_spectro.T, hparams)
save_wav(wav, os.path.join(out_dir, 'test_linear_{}.wav'.format(mel_file.replace('/', '_').replace('\\', '_').replace('.npy', ''))),
        sr=hparams.sample_rate)

plot_spectrogram(lin_spectro, path=os.path.join(out_dir, 'test_linear_{}.png'.format(mel_file.replace('/', '_').replace('\\', '_').replace('.npy', ''))),
                auto_aspect=True)