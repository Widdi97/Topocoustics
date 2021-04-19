import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import get_window
try: #datei muss erst wieder aufgenommen werden
    import responseCleanup
except:
    from utilsLib import responseCleanup

"""
Hier werden zunächst aufnahme und abgespielte datei synchronisiert (Latenzzeit der Audiointerface + der sound-libraries).
Dann wird der Sweep abschnittweise fouriertransformiert und eine Gaußkurve angepasst. Der größte Teil des Codes ist 
dafür da, um zu überprüfen, ob der Gauß-Fit funktioniert hat und ggf. einen erneuten fit an die Daten mit anderen Start-
parametern durchzuführen.

TODO:
    - prüfe, ob abscissa direkt als list erstellt werden kann
    - Code doubling bei der Suche der Startparameter entfernen
"""

class SpectrogramClassifier:
    def __init__(self,binSize,overlap,sampleFreq=96000,window="flattop",micName="",datasetName=""):
        self.dataSet_Name = datasetName
        self.binSize = binSize
        self.overlap = overlap
        self.window = window # Window names: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html
        self.sampleFreq = sampleFreq
        self.micName = micName

    def loadData(self):
        self.timeline = np.load(self.dataSet_Name+"timeline.npy")
#        self.speaker = np.load(self.dataSet_Name+"speaker.npy")
        self.mic = np.load(self.dataSet_Name+"mic.npy")
        self.freq = np.load(self.dataSet_Name+"frequency.npy")
    
    def spectrogram(self):
        #load response function:
        self.loadData()
        if self.window == "":
            window = np.ones(self.binSize)
        else:
            window = get_window(self.window, self.binSize) #test: flat top window
        tSteps = len(self.timeline)
        kMax = tSteps//(self.binSize-self.overlap)
        if (self.binSize-self.overlap)*(kMax+1)+self.overlap >= tSteps: #last bin can't reach further than the length of timeline
            kMax += -1
#        offsets = []
        micAudio = self.mic.T[0]
        abscissa = np.arange(0,self.sampleFreq,self.sampleFreq/self.binSize)
        abscissa = list(abscissa[:len(abscissa)//2])
        #load response function:
        if self.micName == "":
            inverseResponse = np.ones(len(abscissa))
        else:
            cu = responseCleanup.cleanup()
            inverseResponse = cu.inverseResponseFactor(self.micName,abscissa)
        #find freq offset and fit parameters:
        max_latency = 0.5 #[s]
        freq_step = abscissa[1] - abscissa[0]
        freq_maxLatency = ((self.freq[-1] - self.freq[0])/self.timeline[-1])*max_latency
        if freq_maxLatency < 50*freq_step:
            freq_maxLatency = 50*freq_step
        print("freq_maxLatency: ",freq_maxLatency)
        cutOffsets = []
        plotStepsBool = True# __name__=="__main__"
        for k in range(kMax//10,kMax):# only check offset above the lab noise level
            x = micAudio[k*(self.binSize-self.overlap):(k+1)*(self.binSize-self.overlap)+self.overlap]*window
            y = scipy.fft(x)
            y = y[:len(y)//2] #symmetrisierung --> obere Hälfte abgeschnitten
            f_input = self.freq[k*(self.binSize-self.overlap)+self.binSize//2] #input frequency
            cutAbscissa = abscissa[self.getIndex(abscissa,f_input-freq_maxLatency):self.getIndex(abscissa,f_input+freq_maxLatency)]
            cutY = y[self.getIndex(abscissa,f_input-freq_maxLatency):self.getIndex(abscissa,f_input+freq_maxLatency)]
            FreqOfMaxIntens = cutAbscissa[list(np.absolute(cutY)).index(max(np.absolute(cutY)))]
            cutOffsets.append(f_input - FreqOfMaxIntens)
        cutOffsets.sort()
        medFreqOffset = cutOffsets[len(cutOffsets)//2]
        AvgFreqOffset = sum(cutOffsets[len(cutOffsets)//5*1:len(cutOffsets)//5*4])/len(cutOffsets[len(cutOffsets)//5:len(cutOffsets)//5*4])
#        print(cutOffsets)
        latency = medFreqOffset/((self.freq[-1] - self.freq[0])/self.timeline[-1])
        print("medFreqOffset:",medFreqOffset)
        print("AvgFreqOffset:",AvgFreqOffset)
        print("latency: ",str(latency*1000)+"ms")
        self.freqCorrected = self.freq - medFreqOffset
        self.firstHarmonicIntens = []
        self.firstHarmonicIntensGaussianFit = [] #maximum of gauss fit
        self.harmonicsAbscissa = []
        self.harmonicsAbscissa2 = []
        GF = GaussianFit()
        index = 0
        fitParams = []
        coeff = [10,1000,10]
        #==============find fit start params=============
        # first run: best squares  second run: with good starting params
        bestParams = []
        squares = 1e99
        allSquares = []
        for k in [m for m in range(kMax//10,kMax)][::20]:
            x = micAudio[k*(self.binSize-self.overlap):(k+1)*(self.binSize-self.overlap)+self.overlap]*window
            y = scipy.fft(x)
            y = y[:len(y)//2]*inverseResponse #symmetrisierung --> obere Hälfte abgeschnitten
            f_input = self.freqCorrected[k*(self.binSize-self.overlap)+self.binSize//2] #mittelpunkt des bin
            if f_input > 0:
                try:
                    i_0 = self.getIndex(abscissa,f_input-500) #start index
                    i_1 = self.getIndex(abscissa,f_input+500) #end index
                    coeff = GF.optimize(abscissa[i_0:i_1],np.absolute(y)[i_0:i_1],10, f_input,10)
                    if coeff[0] != 10 and coeff[2] != 10:
                        newSquares = self.squares(np.absolute(y)[i_0:i_1],GF.gaussFunction(abscissa[i_0:i_1],*coeff))
                        allSquares.append(newSquares)
                        if newSquares < squares:
                            bestParams = coeff
                            squares = newSquares
                except:
                    pass
        allSquares.sort()
        squares_threshold = allSquares[round(len(allSquares)*4/5)]
        print("squares_threshold: ",squares_threshold)
        for k in [m for m in range(kMax//10,kMax)][::20]:
            x = micAudio[k*(self.binSize-self.overlap):(k+1)*(self.binSize-self.overlap)+self.overlap]*window
            y = scipy.fft(x)
            y = y[:len(y)//2]*inverseResponse #symmetrisierung --> obere Hälfte abgeschnitten
            f_input = self.freqCorrected[k*(self.binSize-self.overlap)+self.binSize//2] #mittelpunkt des bin
            if f_input > 0:
                try:
                    i_0 = self.getIndex(abscissa,f_input-500) #start index
                    i_1 = self.getIndex(abscissa,f_input+500) #end index
                    coeff = GF.optimize(abscissa[i_0:i_1],np.absolute(y)[i_0:i_1],bestParams[0], f_input,bestParams[2])
                    if coeff[0] != bestParams[0] and coeff[2] != bestParams[2]:
                        fitParams.append(coeff)
                except:
                    pass
        fitParams = np.median(np.array(fitParams),axis=0)#np.average(fitParamsArr,axis=0)
        print("fitParams:",fitParams)
        #=======================all Gauss Fits=========================
        lastCoeff = fitParams
        gfIntens = 1
        for k in range(kMax):
            x = micAudio[k*(self.binSize-self.overlap):(k+1)*(self.binSize-self.overlap)+self.overlap]*window
            y = scipy.fft(x)
            y = y[:len(y)//2]*inverseResponse #symmetrisierung --> obere Hälfte abgeschnitten
            f_input = self.freqCorrected[k*(self.binSize-self.overlap)+self.binSize//2] #mittelpunkt des bin
            cutAbscissa = abscissa[self.getIndex(abscissa,f_input-5*fitParams[0]):self.getIndex(abscissa,f_input+5*fitParams[0])]
#            print(cutAbscissa)
            cutY = y[self.getIndex(abscissa,f_input-5*fitParams[0]):self.getIndex(abscissa,f_input+5*fitParams[0])]
            f_input_errorFlag = False
            if f_input > 0:#durch die Frequenzkorrektur (Latenz) kann f_input negativ werden
                lastFreq = 0
                for abscFreq in abscissa[index:]:
                    if lastFreq < f_input and f_input <= abscFreq: #suche frequenz in der FFT abszisse, die kurz über der derzeitigen inputfrequenz liegt
                        index = abscissa.index(abscFreq)
                        self.harmonicsAbscissa.append(f_input)
                        self.firstHarmonicIntens.append(np.absolute(y)[index])
                        break
                    if abscFreq == abscissa[-1]:
                        f_input_errorFlag = True
#                        print("badHarmApp!\nabscFreq: ",abscFreq,"\nf_input: ",f_input,"\nmax(abscissa): ",max(abscissa))
                    lastFreq = abscFreq
                if not f_input_errorFlag:
                    try:
#                        ased = 0/0
                        coeff = GF.optimizeNoBounds(cutAbscissa,np.absolute(cutY),fitParams[0], f_input,fitParams[2])
                        if self.squares(np.absolute(cutY),GF.gaussFunction(cutAbscissa,*coeff)) > squares_threshold:
                            coeff = GF.optimizeNoBounds(cutAbscissa,np.absolute(cutY),2*fitParams[0], f_input,gfIntens*80*fitParams[2])
                        gfIntens = GF.gaussFunction(coeff[1],*coeff)
                        if gfIntens/np.absolute(y)[index] > 5:
                            coeff = GF.optimizeNoBounds(cutAbscissa,np.absolute(cutY),lastCoeff[0], f_input,lastCoeff[2])
                            gfIntens = GF.gaussFunction(coeff[1],*coeff)
                        self.firstHarmonicIntensGaussianFit.append(gfIntens)
                        lastCoeff = coeff
                    except:
                        try:
                            coeff = GF.optimize(cutAbscissa,np.absolute(cutY),fitParams[0], f_input,fitParams[2])
                            if self.squares(np.absolute(cutY),GF.gaussFunction(cutAbscissa,*coeff)) > squares_threshold:
                                coeff = GF.optimize(cutAbscissa,np.absolute(cutY),2*lastCoeff[0], f_input,2*lastCoeff[2])
                            self.firstHarmonicIntensGaussianFit.append(GF.gaussFunction(coeff[1],*coeff))
                            lastCoeff = coeff
                        except:
                            self.firstHarmonicIntensGaussianFit.append(0.00001)
                            lastCoeff = fitParams
        plt.plot(self.harmonicsAbscissa,self.firstHarmonicIntens)#cleanupVals*self.firstHarmonicIntens)
#       ===============================
        plt.plot(self.harmonicsAbscissa,self.firstHarmonicIntensGaussianFit)#cleanupVals*self.firstHarmonicIntensGaussianFit)
        plt.yscale("log")
        plt.title("Intensity of the 1st harmonic")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Intensity")
        plt.xlim((self.freq[0],self.freq[-1]))
#        plt.savefig("intensity_1st_harmonic.pdf")
        np.save("spectrogram",np.array([self.harmonicsAbscissa,self.firstHarmonicIntens,self.firstHarmonicIntensGaussianFit]))
        plt.show()
        
    def getIndex(self,sortedArray,val): #returns the index at which val is bigger than an item from 1d sortedArray for the first time
        if sortedArray[0] > val:
            return 0
        for v in sortedArray:
            if v > val:
                return list(sortedArray).index(v)-1
        return len(sortedArray)-1
    
    def squares(self,array1,array2):
        return np.sum((array2-array1)**2)
        

class GaussianFit:
    def __init__(self):
        pass
    
    def gaussFunction(self,x,sigma,mu,A):
        return A/(sigma*(2.0*np.pi)**0.5)*np.exp(-(x-mu)**2/(2.0*sigma**2))

    def optimize(self,xVals,yVals,sigma0,mu0,A0):
        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        p0 = [sigma0, mu0, A0]
        bounds = ([0,mu0-2*sigma0,0],[sigma0*1e5,mu0+2*sigma0,np.inf]) #([0,mu0-2*sigma0,0],[sigma0*1e5,mu0+2*sigma0,np.inf])
        coeff, var_matrix = scipy.optimize.curve_fit(self.gaussFunction,xVals,yVals,p0=p0,bounds=bounds)
        return coeff
    
    def optimizeNoBounds(self,xVals,yVals,sigma0,mu0,A0):
        p0 = [sigma0, mu0, A0]
        coeff, var_matrix = scipy.optimize.curve_fit(self.gaussFunction,xVals,yVals,p0=p0)
        return coeff

if __name__ == "__main__":
    sc = SpectrogramClassifier(1323,1323//2,44100,"flattop")
    sc.spectrogram()
        
        
        
        
        
        
        