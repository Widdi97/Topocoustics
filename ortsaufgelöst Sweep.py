from utilsLib import audio, spectrogram
import os, os.path
import shutil
import time

def record_localized_sweep(N,sweepTime=60,f_0=100,f_1=20000,sampleRate=96000):
    filenames = ["frequency.npy","mic.npy","mic2.npy","speaker.npy","timeline.npy","spectrogram.npy"]
    dirname, filename = os.path.split(os.path.abspath(__file__))
    a = audio.AudioSweep(f_0,f_1,sweepTime,sampleRate,False,False)#16250
    a.play()
    time.sleep(3)
    sc = spectrogram.SpectrogramClassifier(sweepTime*sampleRate//2000,sweepTime*sampleRate//2000//2,sampleRate,"flattop","bd31004")
    sc.spectrogram()
    os.makedirs(os.path.join(dirname,"measurement\\"+str(N)))
    for fname in filenames:
        try:
            shutil.move(dirname+"\\"+fname, dirname+"\\measurement\\"+str(N)+"\\"+fname)
        except:
            print("err - measurement already exists!")
            return

record_localized_sweep(5,60)