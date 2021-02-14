import numpy as np
from scipy.signal import correlate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from utilsLib import audio
import scipy
import time
from scipy.signal import get_window

def adfg(t,phi,w,A):
    a = w*t+phi
    return A*np.sin(a)

class phaseCalc: # returns the sample offset of highest correlation
    def __init__(self):
        self.w = 1 # placeholder for initiation

    def sin(self,t,phi,w):#,w):
        return 4.52*np.sin(w*t+phi)
    
    def phaseShift(self,t,array1,array2,freq,sampleRate):
        t = t[:50000]
        array1 = array1[:50000]
        array2 = array2[:50000]
        self.w = 2*np.pi*freq
        samples_per_2pi = sampleRate/freq
#        plt.plot(t,array1)
#        plt.plot(t,array2)
##        plt.xlim((1,1.0005))
#        plt.show()
        corr = correlate(array1,array2,mode="same")
        sampleOffset = [tme/sampleRate - len(corr)/2/sampleRate for tme in range(len(corr))]
        fitData = corr[len(corr)//2 - 100:len(corr)//2 + 100]
        fitAbsc = sampleOffset[len(corr)//2 - 100:len(corr)//2 + 100]
        params1, pcov1 = curve_fit(np.sin,t,array1)
        print(params1)
        plt.plot(fitAbsc,fitData)
        plt.plot(fitAbsc,adfg(np.array(fitAbsc),*params1))
#        plt.xlim((-100,100))
        plt.show()
        return 1
        
    def measurement(self,freq,sampleRate):
#        a = audio.AudioSweep(freq,freq,5,sampleRate,False,False)
#        a.play()
#        time.sleep(1)
        mic = np.load("mic.npy").T[0][sampleRate*2//2:][0:50000]
        mic2 = np.load("mic2.npy").T[0][sampleRate*2//2:][0:50000]
        timeline = np.load("timeline.npy")[sampleRate*2//2:][0:50000]
        print("len(timeline): ",len(timeline))
        # plt.plot(timeline,mic)
        # plt.plot(timeline,mic2)
        # plt.show()
        p = self.phaseShift(timeline,mic,mic2,freq,sampleRate)
#        print(p)
        return p


if __name__ == "__main__": # test
    ph = phaseCalc()
    # ph.measurement(2248, 96000)
    phases = []
    for fr in [7750]:#[5350 + 10*k for k in range(0,250,10)]:#[293,536,786,1062,1301,1537,1755,2250,2390,2608,2844,3080,3312,3530,3711,4473,4556,4734,4945,5162,5375,5568,5717,6683,6746,6894,7081,7280,7476,7651,7778]:
        phases.append(ph.measurement(fr, 96000))
    # copy-paste in excel:
    st = ""
    for k in phases:
        st += str(k)+"\t"
    print(st)
    # mic = np.load("mic.npy").T[0][96000*2//2:][0:1000]
    # mic2 = np.load("mic2.npy").T[0][96000*2//2:][0:1000]
    # timeline = np.load("timeline.npy")[96000*2//2:][0:1000]
    # ph = phaseCalc()
    # p = ph.phaseShift(timeline,mic,mic2,3708)