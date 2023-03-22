import pygad.kerasga
import pygad


import numpy as np
from kapibara_audio import BUFFER_SIZE,SPECTOGRAM_WIDTH
from emotions import EmotionTuple

import tensorflow as tf
import time

from tensorflow.keras import layers
from tensorflow.keras import models

import math
import os.path

from timeit import default_timer as timer
from tflitemodel import LiteModel


DISTANCE_MAX_VAL=2048.0 # in mm
MAX_ANGEL=360.0

FLOOR_SENSORS_COUNT=8
FRONT_SENSORS_COUNT=8

class MindOutputs:

    MAX_INPUT_VALUE=4294967295.0

    def __init__(self,speedA:int=0,speedB:int=0,directionA:int=0,directionB:int=0) -> None:
        self.speedA=speedA
        self.directionA=directionA

        self.speedB=speedB
        self.directionB=directionB

    def error(self)->float:
        error=0

        if self.speedA>100:
            error-=50*(self.speedA/100.0)
        if self.speedB>100:
            error-=50*(self.speedB/100.0)
        if self.directionA>3:
            error-=50*(self.directionA/3.0)
        if self.directionB>3:
            error-=50*(self.directionB/3.0)

        if math.isnan(self.speedA):
            error-=100
        if math.isnan(self.speedB):
            error-=100
        if math.isnan(self.directionA):
            error-=100
        if math.isnan(self.directionB):
            error-=100

        return error

    def get(self)->list[float]:
        return [self.speedA,self.speedB,self.directionA,self.directionB]
    
    def get_norm(self)->list[float]:
        return [self.speedA/100.0,self.speedB/100.0,self.directionA/4.0,self.directionB/4.0]
    
    def set_from_norm(self,speedA:float,speedB:float,directionA:float,directionB:float):

        self.speedA=speedA*100
        self.speedB=speedB*100

        if directionA>=0 and directionA<0.25:
            self.directionA=0
        elif directionA>=0.25 and directionA<0.5:
            self.directionA=1
        elif directionA>=0.5 and directionA<0.75:
            self.directionA=2
        else:
            self.directionA=3
        
        if directionB>=0 and directionB<0.25:
            self.directionB=0
        elif directionB>=0.25 and directionB<0.5:
            self.directionB=1
        elif directionB>=0.5 and directionB<0.75:
            self.directionB=2
        else:
            self.directionB=3

        if self.speedA>self.MAX_INPUT_VALUE:
            self.speedA=self.MAX_INPUT_VALUE

        if self.speedB>self.MAX_INPUT_VALUE:
            self.speedB=self.MAX_INPUT_VALUE

        if self.directionA>self.MAX_INPUT_VALUE:
            self.directionA=self.MAX_INPUT_VALUE

        if self.directionB>self.MAX_INPUT_VALUE:
            self.directionB=self.MAX_INPUT_VALUE

    def motor1(self)->tuple[int,int]:
        return (int(self.speedA),int(self.directionA))
    
    def motor2(self)->tuple[int,int]:
        return (int(self.speedB),int(self.directionB))

class Mind:
    OUTPUTS_BUFFER=10
    NUM_GENERATIONS=50

    '''A class that represents decision model
    A inputs: 
    *gyroscope
    *accelerometer
    *distance sensors:
        -front
        -floor
    *audio - mono,combined stereo
    *audio coefficent, wich audio channel is stronger
    *10 last outputs

    A outputs:


    Outputs:

    '''
    

    def __init__(self,emotions:EmotionTuple) -> None:
        self.last_outputs=np.array([MindOutputs(0,0,0,0)]*10)

        self.gyroscope=np.zeros(3,dtype=np.float32)
        self.accelerometer=np.zeros(3,dtype=np.float32)

        self.dis_front=0.0

        self.dis_floor=0.0

        self.audio=np.zeros(BUFFER_SIZE,dtype=np.float32)

        self.front_sensors=np.zeros(FRONT_SENSORS_COUNT,dtype=np.float32)

        self.floor_sensors=np.zeros(FLOOR_SENSORS_COUNT,dtype=np.float32)

        self.audio_coff=(0,0)

        self.emotions=emotions

        ''' 0...2 gyroscope, 3...5 accelerometer, 6...7 audio coefficient, floor_sensors, 
        front_sensors, spectogram'''

        self.inputs=np.ndarray(len(self.gyroscope)+len(self.accelerometer)+SPECTOGRAM_WIDTH**2+
                               len(self.audio_coff)+FLOOR_SENSORS_COUNT+FRONT_SENSORS_COUNT,dtype=np.float32)
        
        #self.inputs=self.inputs.reshape(len(self.inputs),1)

    
    def init_model(self):

        input=tf.keras.layers.Input([None,len(self.inputs)])

        layer1=tf.keras.layers.LSTM(512,return_sequences=True)(input)

        layer2=tf.keras.layers.LSTM(256,return_sequences=True)(layer1)

        output=tf.keras.layers.Dense(4,activation='relu')(layer2)

        self.model=tf.keras.Model(inputs=input,outputs=output)



    def run_model(self,solutions):

        self.prepareInput()

        predictions=pygad.kerasga.predict(model=self.model,
                        solution=solutions,
                        data=self.inputs.reshape(1,len(self.inputs)))
                
        return predictions
        
        

    def getData(self,data:dict):
        
        self.gyroscope:np.array=data["Gyroscope"]["gyroscope"]/MAX_ANGEL
        self.accelerometer:np.array=data["Gyroscope"]["acceleration"]/DISTANCE_MAX_VAL

        self.front_sensors[0]=data["Distance_Front"]["distance"]/8160.0

        self.floor_sensors[0]=data["Distance_Floor"]["distance"]/8160.0

        left:np.array=np.array(data["Ears"]["channel1"],dtype=np.float32)/32767.0
        right:np.array=np.array(data["Ears"]["channel2"],dtype=np.float32)/32767.0

        self.audio:np.array=np.add(left,right,dtype=np.float32)/2.0

        m:float=np.mean(self.audio)

        l:float=np.mean(left)
        r:float=np.mean(right)

        if m==0:
            self.audio_coff=(0.0,0.0)
            return

        self.audio_coff=(l/m,r/m)

    def prepareInput(self,spectogram):
        
        np.put(self.inputs,0,self.gyroscope)
        np.put(self.inputs,3,self.accelerometer)
        np.put(self.inputs,6,self.audio_coff)
        np.put(self.inputs,8,self.front_sensors)
        np.put(self.inputs,8+len(self.floor_sensors),self.floor_sensors)


    '''Do wyjebania'''
    def push_output(self,output:MindOutputs):
        
        self.last_outputs=np.roll(self.last_outputs,-1)
        self.last_outputs[-1]=output

    def loop(self):
        pass
        