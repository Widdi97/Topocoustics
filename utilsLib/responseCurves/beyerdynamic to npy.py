import numpy as np
import matplotlib.pyplot as plt
import csv
#
files = ["449350_31004_0Grad","449350_31004_90Grad",
         "449350_31047_0Grad","449350_31047_90Grad"]



def dB_kurven_verwursten():
    global files
    for k in range(len(files)):
        data = []
        with open(files[k]+".txt", 'r') as file:
            reader = csv.reader(file,delimiter="\t")
            for row in reader:
                data.append(row)
        data = np.array(data[2:],dtype=np.float).T
        plt.plot(data[0],data[1])
        plt.title("response curve for "+files[k])
        plt.ylabel("Response Curve [dB]")
        plt.xlabel("Frequency [Hz]")
        plt.show()
        amplitude = 10**(data[1]/20) # data ist in dB gegeben
        #response = np.load("/responseCurv
        plt.plot(data[0],amplitude)
        plt.title("response curve for "+files[k])
        plt.ylabel("Response Curve [a.u.]")
        plt.xlabel("Frequency [Hz]")
        plt.show()
        data2 = np.array([data[0],amplitude])
        np.save(files[k]+"_abs_amplitude",data2)

dB_kurven_verwursten()