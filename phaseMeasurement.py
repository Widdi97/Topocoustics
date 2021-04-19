import numpy as np
from scipy.optimize import  curve_fit
import matplotlib.pyplot as plt
from utilsLib import audio
import scipy
import time

"""
performs measurement, fits sin-curves and calculates the relative phase between l
eft and right mic input for specified frequencies
"""

class phaseCalc:
    def __init__(self):
        self.w = 1 # placeholder for initiation
    
    def sin(self,t,phi,A):
        return A*np.sin(self.w*t+phi)
    
    def phaseShift(self,t,array1,array2,freq):
        t = t[:1000]
        array1 = array1[:1000]
        array2 = array2[:1000]
        self.w = 2*np.pi*freq
        bounds = ([0,0],[2*np.pi,np.inf])
        params1, pcov1 = curve_fit(self.sin,t,array1,bounds=bounds)
        params2, pcov2 = curve_fit(self.sin,t,array2,bounds=bounds)
        print(params1,params2)
        perr1 = np.sqrt(np.diag(pcov1))
        perr2 = np.sqrt(np.diag(pcov2))
        error = (perr1[0]**2 + perr2[0]**2)**0.5
        print("error:",error,"\nerror degrees:",error*360/2/np.pi)
        plt.plot(t,array1)
        plt.plot(t,array2)
        plt.xlim((1,1.0005))
        return params2[0] - params1[0]
    
    def phaseShift2(self,t,array1,array2,freq):
        rSample = round(len(t)/(t[-1]-t[0]))
        freq_axis = np.linspace(0,rSample/2,len(array1)//2)
        index = len(freq_axis)
        for f in freq_axis:
            if freq < f:
                index = list(freq_axis).index(f)-1
                break
        print(index,freq_axis[index])
        global y1
        global y2
        y1 = scipy.fft(array1)
        y2 = scipy.fft(array2)
        y1 = y1[:len(y1)//2] #symmetrisierung --> obere Hälfte abgeschnitten
        y2 = y2[:len(y2)//2] 
        plt.plot(t,array1)
        plt.plot(t,array2)
        plt.xlim((1,1.0005))
#        plt.xlim((1,1.001))
        plt.show()
        angle1 = np.angle(y1[index])
        angle2 = np.angle(y2[index])
#        print(angle1,angle2)
        print(angle2 - angle1)
        return angle2 - angle1
        
    def measurement(self,freq,sampleRate):
        a = audio.AudioSweep(freq,freq,1.5,sampleRate,False,False)
        a.play()
        time.sleep(0.8)
        mic = np.load("mic.npy").T[0][sampleRate*2//2:][0:50000]
        mic2 = np.load("mic2.npy").T[0][sampleRate*2//2:][0:50000]
        timeline = np.load("timeline.npy")[sampleRate*2//2:][0:50000]
        # plt.plot(timeline,mic)
        # plt.plot(timeline,mic2)
        # plt.show()
        p = self.phaseShift2(timeline,mic,mic2,freq)
#        print(p)
        return p


if __name__ == "__main__": # test
    ph = phaseCalc()
    # ph.measurement(2248, 96000)
    phases = []
    for fr in [293,536,786,1062,1301,1537,1755,2250,2390,2608,2844,3080,3312,3530,3711,4473,4556,4734,4945,5162,5375,5568,5717,6683,6746,6894,7081,7280,7476,7651,7778]:
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