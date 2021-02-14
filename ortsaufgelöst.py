from utilsLib import audio
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import get_window


class freqTest:
    def __init__(self,f_test,sweepTime=5,window="flattop",sampleRate=96000):
        self.f_test = f_test
        self.sweepTime = sweepTime
        self.window = window
        self.sampleRate = sampleRate
        
    def run(self):
        plotStepsBool = True
        a = audio.AudioSweep(self.f_test,self.f_test,self.sweepTime,self.sampleRate,False,False)
        a.play()
        mic = np.load("mic.npy")
        timeline = np.load("timeline.npy")
        mic2 = mic.T[0]
        window = get_window(self.window, len(mic2))
        y = scipy.fft(mic2*window)
        y = y[:len(y)//2]
        y = np.abs(y)
        abscissa = np.array([k/timeline[-1] for k in range(int(round(timeline[-1])*96000//2))])
        inputIndex = self.getIndex(abscissa,self.f_test)
        testIntensity = (y[inputIndex]+y[inputIndex+1])/2 # nächstkleinere und größere Stelle um den zu testenden Wert
        print("Intensity at {}Hz:\n".format(self.f_test),testIntensity)
        if plotStepsBool:
            plt.plot(abscissa,y)
            plt.yscale("log")
            plt.scatter([self.f_test],[testIntensity],color="red")
            plt.xlim(self.f_test-20,self.f_test+20)
            plt.show()

    def getIndex(self,sortedArray,val): #returns the index at which val is bigger than an item from 1d sortedArray for the first time
        if sortedArray[0] > val:
            return 0
        for v in sortedArray:
            if v > val:
                return list(sortedArray).index(v)-1
        return len(sortedArray)-1

ft = freqTest(13518)
ft.run()