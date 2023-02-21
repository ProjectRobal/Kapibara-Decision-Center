from emotions import EmotionModifier,EmotionTuple
from kapibara_audio import KapibaraAudio,BUFFER_SIZE
import numpy as np
import time

class HearingCenter(EmotionModifier):
    '''modifiers with KapibaraAudio model'''
    def __init__(self) -> None:
        super().__init__()
        self.hearing=KapibaraAudio('./hearing')
        self.audio=np.zeros(BUFFER_SIZE,np.int16)
        # a vector of x/m where x is mean value of channel and m is mean value of added signals
        #self.coefficient=(0,0)
    
    def retriveData(self,data:dict):
        try:
            '''get a specific data from host'''
            left:np.array=np.array(data["Ears"]["channel1"],dtype=np.float32)/32767.0
            right:np.array=np.array(data["Ears"]["channel2"],dtype=np.float32)/32767.0

            self.audio:np.array=np.add(left,right,dtype=np.float32)/2

            #m:float=np.mean(self.audio,dtype=np.float32)
            #l:float=np.mean(left,dtype=np.float32)
            #r:float=np.mean(right,dtype=np.float32)

            #self.coefficient=(l/m,r/m)

        except:
            print("Audio data is missing!")


    def modify(self,emotions:EmotionTuple):
    
        output=self.hearing.input(self.audio)

        if output=="unsettling":
            emotions.unsettlement=1
        elif output=="pleasent":
            emotions.pleasure=1
        elif output=="scary":
            emotions.fear=1
        elif output=="nervous":
            emotions.anger=1


class FrontSensorModifier(EmotionModifier):
    def __init__(self,name:str) -> None:
        '''name - sensor name'''
        super().__init__()
        # a distance in wich robot will 'feel' pain in mm
        self.THRESHOLD=50
        self.distance=0
        self.name=name

    def retriveData(self, data: dict):
        try:
            self.distance=data[self.name]["distance"]
        except:
            print("Cannot get data from sensor: ",self.name)
    
    def modify(self, emotions: EmotionTuple):
        
        if self.distance <= self.THRESHOLD:
            emotions.fear=1

class FloorSensorModifier(EmotionModifier):
    def __init__(self,name:str) -> None:
        '''name - sensor name'''
        super().__init__()
        # a distance in wich robot will 'feel' pain in mm
        self.THRESHOLD=100
        self.distance=0
        self.name=name

    def retriveData(self, data: dict):
        try:
            self.distance=data[self.name]["distance"]
        except:
            print("Cannot get data from sensor: ",self.name)
    
    def modify(self, emotions: EmotionTuple):
        
        if self.distance >= self.THRESHOLD:
            emotions.fear=1


class ShockModifier(EmotionModifier):
    '''A modifier that use gyroscope and accelerometer data to detect drag'''
    def __init__(self) -> None:
        super().__init__()
        self.last_time=0
        self.gyroscope=np.zeros(3,dtype=np.float32)
        self.last_gyroscope=np.zeros(3,dtype=np.float32)
        self.acceleration=np.zeros(3,dtype=np.float32)
        self.last_acceleration=np.zeros(3,dtype=np.float32)

        self.GYRO_THRESHOLD=400000
        self.ACCEL_THRESHOLD=200000

    def retriveData(self, data: dict):
        try:
            self.gyroscope=np.array(data["Gyroscope"]["gyroscope"],dtype=np.float32)
            self.acceleration=np.array(data["Gyroscope"]["acceleration"],dtype=np.float32)
        except:
            print("Cannot retrive data from IMU!")

    def modify(self, emotions: EmotionTuple):

        self.time=time.time()

        if self.last_time==0:
            self.last_time=time.time()

        if np.mean(self.last_gyroscope)>0:
            drag=np.subtract(self.last_gyroscope,self.gyroscope,dtype=np.float32)/(self.time-self.last_time)

            if np.mean(drag)>self.GYRO_THRESHOLD:
                print(np.mean(drag))
                emotions.anger=1

        if np.mean(self.last_acceleration)>0:
            drag=np.subtract(self.last_acceleration,self.acceleration,dtype=np.float32)/(self.time-self.last_time)

            if np.mean(drag)>self.ACCEL_THRESHOLD:
                emotions.fear=1
        
        self.last_gyroscope=self.gyroscope
        self.last_acceleration=self.acceleration

        self.last_time=self.time
