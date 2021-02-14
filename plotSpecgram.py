import matplotlib.pyplot as plt
import numpy as np

specgram = np.load("spectrogram.npy")
freq = np.load("frequency.npy")

#plt.plot(specgram[0],specgram[1])
plt.plot(specgram[0],specgram[2])
plt.yscale("log")
#plt.title("Intenstät der ersten Harmonischen in SSH Konfiguration")
plt.xlabel("Frequenz [Hz]")
plt.ylabel("Intensität [a.u.]")
plt.xlim((100,2e4))
plt.show()