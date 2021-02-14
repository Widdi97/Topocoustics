import sounddevice as sd #recording audio
import simpleaudio as sa #playing audio
from scipy.io.wavfile import write
import time
import numpy as np
from threading import Thread


""" 71 aktive Zeilen
Simple utility for playing and recording audio.

TODO:
    - Finished - flags in allen Threads implenmentieren und Threads joinen

Naming convention:                                                   Data type
    - speaker audio: {}speaker.wav and speaker_{}.npy               np-Array (1xN)
    - microphone audio: {}mic.wav and mic_{}.npy                    np-Array (Nx2) ! --> ACHTUNG beim auswerten:  dim((Nx2).T) = (2xN)
    - time steps: {}timeline.npy                                    np-Array (1xN)
    - frequency (speaker): {}frequency.npy                          np-Array (1xN)
    
    {} = "{}Hz to {}Hz, {}s, linear={} - ".format(self.f_0,self.f_1,self.seconds,self.exponentialBool)
    
    - the device-speciffic offset of mic and speaker has to be set for every soundcard/computer
        --> lin-sweep measurement --> 
"""


class AudioSweep:
    def __init__(self,f_0,f_1,deltaT,sampleRate=44100,exponentialSweep=False,autoNaming=False):
        self.SampleRate = sampleRate #recording sample rate [Hz]
        self.f_0 = f_0 # starting frequency [Hz]
        self.f_1 = f_1 # ending frequency [Hz]
        self.seconds = deltaT # sweep time [s]
        self.currentFreq = 500# used for communication between sweepGen and playHarmonic [Hz]
        self.exponentialBool = exponentialSweep
        self.recordBackgroundBool = False
        if autoNaming:
            self.fileNamePrefix = "{}Hz to {}Hz, {}s, linear={} - ".format(self.f_0,self.f_1,self.seconds,self.exponentialBool)
        else:
            self.fileNamePrefix = ""
        if __name__=="__main__":
            self.play()
        
    def sweepGen(self):
        self.timeline = np.linspace(0, self.seconds, self.seconds * self.SampleRate, False) #float in s
        if not self.exponentialBool:
            self.f_t = 2 * self.f_0 + (self.f_1 - self.f_0)/self.seconds*self.timeline # = self.f_0 + (self.f_1 - self.f_0)/self.seconds*self.timeline
        else:
            self.f_t = self.f_0*(self.f_1/self.f_0)**(self.timeline/self.seconds)
        note = np.sin(self.f_t * self.timeline * np.pi) # np.sin(self.f_t * self.timeline * 2 * np.pi)
        self.f_t = self.f_t.astype(np.float32)
        self.timeline = self.timeline.astype(np.float32)
        # Ensure that highest value is in 16-bit range
        audio = note * (2**15 - 1) / np.max(np.abs(note))
        audio = audio.astype(np.float32)
        np.save(self.fileNamePrefix+"speaker",audio)
        # Convert to 32-bit data
        audio = audio.astype(np.int16)
        # Start playback
        
        
        
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
        
        
        
        
        
        play_obj = sa.play_buffer(audio, 1, 2, self.SampleRate) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  bytes per sample auf 1 setzen
        # Wait for playback to finish before exiting
        play_obj.wait_done()
#        write(self.fileNamePrefix+'speaker.wav', self.SampleRate, audio)  # Save as WAV file
#        np.save(self.fileNamePrefix+"speaker",audio)
        np.save(self.fileNamePrefix+"timeline",self.timeline)
        np.save(self.fileNamePrefix+"frequency",self.f_t-self.f_0)
    
    def recorder(self): #exports float32 array
        myrecording = sd.rec(int(self.seconds * self.SampleRate), samplerate=self.SampleRate, channels=2)
#        myrecording = myrecording.astype(np.int16)
#        myrecording=(myrecording>>16).astype(np.int16) 
#        myrecording=(myrecording/(2**32)*2**8).astype(np.int8)
        sd.wait()  # Wait until recording is finished
#        write(self.fileNamePrefix+'mic.wav', self.SampleRate, myrecording)  # Save as WAV file
        np.save(self.fileNamePrefix+"mic",myrecording[:,0:1])
        np.save(self.fileNamePrefix+"mic2",myrecording[:,1:2])
        
    def recordBackground(self):
        myrecording = sd.rec(int(self.seconds * self.SampleRate), samplerate=self.SampleRate, channels=1)
#        myrecording = myrecording.astype(np.int16)
#        myrecording=(myrecording>>16).astype(np.int16) 
#        myrecording=(myrecording/(2**32)*2**8).astype(np.int8)
        sd.wait()  # Wait until recording is finished
#        write(self.fileNamePrefix+'background.wav', self.SampleRate, myrecording)  # Save as WAV file
        np.save(self.fileNamePrefix+"background",myrecording)
            
    def play(self):
        if self.recordBackgroundBool:
            self.recordBackground()
        threads = [
                Thread(target=self.sweepGen),
                Thread(target=self.recorder)
                ]
        for thread in threads:
            thread.start()
        time.sleep(self.seconds+2)
        for thread in threads:
            thread.join()
        print("recording done")

#class AudioSync:
#    def __init__(self):
#        pass

if __name__=="__main__":
    a = AudioSweep(500,5000,2,44100,False,False)# f_0, f_1, deltaT, exponentialMode, auto-naming
    speaker = np.load("speaker.npy")
    mic = np.load("mic.npy")
    mic2 = np.load("mic2.npy")
    timeline = np.load("timeline.npy")
    frequ = np.load("frequency.npy")