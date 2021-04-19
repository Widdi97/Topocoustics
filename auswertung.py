from utilsLib import audio, spectrogram#, responseCleanup
import numpy as np
import matplotlib.pyplot as plt
import time



def visualize():
    NFFT = len(timeline)//1000
    pxx,  freq, t, cax = plt.specgram(np.average(mic.T,axis=0), Fs=sampleRate, NFFT=NFFT,
                                      noverlap=NFFT//2, cmap=plt.cm.inferno,mode='psd',
                                      xextent=(frequ[0],frequ[-1]),interpolation="gaussian")
    plt.xlabel("Speaker frequency [Hz]")
    plt.ylabel("Microphone frequency [Hz]")
    plt.colorbar(cax).set_label('Power spectral density [dB]')
#    plt.xlim(3500,4000)
    plt.xlim(frequ[0],frequ[-1])
#    plt.title("FFT heatmap")
#    plt.savefig("heatmap.pdf")
    plt.show()
    #======================================================


sweepTime = 15 # [s]
sampleRate = 44100#96000#192000,88200,44100 # Wichtig: Muss auch in der Audio-Systemsteuerung eingestellt werden!


a = audio.AudioSweep(100,20000,sweepTime,sampleRate,False,False)#16250
a.play()

print("time: ",time.time())
speaker = np.load("speaker.npy")
mic = np.load("mic.npy")
timeline = np.load("timeline.npy")
frequ = np.load("frequency.npy")
#background = np.load("background.npy")
visualize()
if __name__ == "__main__":
    t0 = time.time()
    sc = spectrogram.SpectrogramClassifier(len(timeline)//1000,len(timeline)//1000//2,sampleRate,"flattop","bd31047")
    sc.spectrogram()
    print("runtime",time.time()-t0)
    del spectrogram