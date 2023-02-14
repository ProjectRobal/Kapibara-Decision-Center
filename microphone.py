import numpy as np

import pyaudio

class Microphone:
    def __init__(self,format=pyaudio.paInt16,channels=1,rate=16000,chunk=4096):
        self.audio=pyaudio.PyAudio()
        self.rate=rate
        self.chunk=chunk
        self.format=format
        self.stream=self.audio.open(format=format,channels=channels,frames_per_buffer=chunk,rate=rate,input=True)

    def record(self,seconds):
        frames=np.array([],dtype=np.int16)

        for i in range(int(self.rate/self.chunk * seconds)):
            data=self.stream.read(self.chunk,exception_on_overflow=False)

        #Conversion from bytes array to samples array
        
            frames=np.append(frames,np.fromstring(data,dtype=np.int16))

        return frames

    def __del__(self):
        self.stream.stop_stream()
        self.stream.close()


