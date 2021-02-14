import numpy as np
import matplotlib.pyplot as plt


"""
Die Klasse soll den Dateinamen der Responsekurve übergeben bekommen
und ein numpy array ausgeben, das die Amplitude in den benötigten
Frequenzschritten ausgibt.
"""

class cleanup:
    def __init__(self):
        if __name__ == "__main__":
            path = "responseCurves/"
        else:
            path = "utilsLib/responseCurves/"
        self.deviceNames = {"bd31004":path+"449350_31004_0Grad_abs_amplitude.npy",
                            "bd31047":path+"449350_31047_0Grad_abs_amplitude.npy"}
        
    def inverseResponseFactor(self,deviceName,freqAxis):
        responseCurve = np.load(self.deviceNames[deviceName])
        return np.interp(freqAxis, responseCurve[0], 1/responseCurve[1])

if __name__ == "__main__":
    c=cleanup()
    testAbscissa = np.arange(0,20000,1)
    plt.scatter(testAbscissa,c.inverseResponseFactor("bd31047",testAbscissa))
#    plt.yscale("log")
    plt.show()