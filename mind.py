
import numpy as np
from enum import Enum
from kapibara_audio import BUFFER_SIZE,SPECTOGRAM_WIDTH
from emotions import EmotionTuple

import tensorflow as tf
import time

from KCN.network import Network
from KCN.layer import RecurrentLayer
from KCN.activation.sigmoid import Sigmoid
from KCN.activation.linear import Linear
from KCN.activation.relu import Relu

import math
import os.path

from timeit import default_timer as timer
import threading
from multiprocessing import Process
import copy

import cv2


'''
Gyroscope is now change of rotation,
Accelration is now a change of position

'''

DISTANCE_MAX_VAL=2048.0 # in mm
MAX_ANGEL=360.0

FLOOR_SENSORS_COUNT=1
FRONT_SENSORS_COUNT=2

class MindOutputs:

    def __init__(self,speedA:int=0,speedB:int=0,directionA:int=0,directionB:int=0,reward:float=0) -> None:
        self.speedA=speedA
        self.directionA=directionA

        self.speedB=speedB
        self.directionB=directionB
        self.reward=reward

    def get(self)->tuple[float]:
        return (self.speedA,self.speedB,self.directionA,self.directionB)
    
    def get_norm(self)->tuple[float]:
        return (self.speedA/100.0,self.speedB/100.0,self.directionA/4.0,self.directionB/4.0)
    
    def get_reward(self)->float:
        return self.reward
    
    def set_reward(self,reward:float):
        self.reward=reward
    
    def set_from(self,speedA:float,speedB:float,directionA:float,directionB:float):

        self.speedA=speedA*100.0
        self.speedB=speedB*100.0

        if directionA>3:
            self.directionA=directionA
        else:
            self.directionA=3
        
        if directionB>3:
            self.directionB=directionB
        else:
            self.directionB=3

    def motor1(self)->tuple[int,int]:
        return (int(self.speedA),int(self.directionA))
    
    def motor2(self)->tuple[int,int]:
        return (int(self.speedB),int(self.directionB))
    
class MindFrame:
    '''A pair of inputs and outputs of mind'''
    def __init__(self,inputs,spectogram,output:MindOutputs,reward:float) -> None:
        self.input=(spectogram,inputs)

        self.output=output
        self.reward=reward

    def getInput(self):
        return self.input
    
    def getOutput(self):
        return self.output
    
    def getReward(self):
        return self.reward
                

class Mind:

    '''A class that represents decision model
    A inputs: 
    *gyroscope
    *accelerometer
    *distance sensors:
        -front
        -floor
    *audio - mono,combined stereo
    *audio coefficent, wich audio channel is stronger
    A outputs:


    Outputs:

    '''

    def __init__(self,emotions:EmotionTuple) -> None:

        self.gyroscope=np.zeros(3,dtype=np.float32)
        self.accelerometer=np.zeros(3,dtype=np.float32)

        self.audio=np.zeros(BUFFER_SIZE,dtype=np.float32)

        self.front_sensors=np.zeros(FRONT_SENSORS_COUNT,dtype=np.float32)

        self.floor_sensors=np.zeros(FLOOR_SENSORS_COUNT,dtype=np.float32)

        self.audio_coff=(0,0)

        self.emotions=emotions

        ''' 0...2 gyroscope, 3...5 accelerometer, 6...7 audio coefficient, floor_sensors, 
        front_sensors, spectogram, 2 outputs from second network'''

        self.input_size=len(self.gyroscope)+len(self.accelerometer)+len(self.audio_coff)+FLOOR_SENSORS_COUNT+FRONT_SENSORS_COUNT+4

        self.inputs=np.ndarray(self.input_size,dtype=np.float32)
        self.last_left_output=np.zeros(4,dtype=np.float32)
        self.last_right_output=np.zeros(4,dtype=np.float32)

        self.left_network=Network(self.input_size)

        self.left_network.addLayer(256,16,RecurrentLayer)
        self.left_network.addLayer(4,8,RecurrentLayer,[Sigmoid,Sigmoid,Sigmoid,Relu])

        self.right_network=Network(self.input_size)

        self.right_network.addLayer(256,16,RecurrentLayer)
        self.right_network.addLayer(4,8,RecurrentLayer,[Sigmoid,Sigmoid,Sigmoid,Relu])
        
        self.spectogram=None

    def getData(self,data:dict):

        self.inputs[0:3]=data["Gyroscope"]["gyroscope"]/MAX_ANGEL
        self.inputs[4:7]=data["Gyroscope"]["acceleration"]/DISTANCE_MAX_VAL

        self.inputs[8]=data["Distance_Front"]["distance"]/8160.0
        self.inputs[9]=data["Distance_Front1"]["distance"]/8160.0

        self.inputs[10]=data["Distance_Floor"]["distance"]/8160.0

        left:np.array=np.array(data["Ears"]["channel1"],dtype=np.float32)/32767.0
        right:np.array=np.array(data["Ears"]["channel2"],dtype=np.float32)/32767.0

        self.audio:np.array=np.add(left,right,dtype=np.float32)/2.0

        m:float=np.max(self.audio)

        l:float=np.max(left)
        r:float=np.max(right)

        if m==0:
            self.audio_coff=(0.0,0.0)
        else:
            self.audio_coff=(l/m,r/m)

        self.inputs[11]=self.audio_coff[0]
        self.inputs[12]=self.audio_coff[1]

        self.spectogram=data["spectogram"]

        self.inputs[13:]=self.spectogram.resize(32*32,)

        
    def loop(self)->MindOutputs:

        self.inputs[self.input_size-2:]=self.last_right_output[:]
        self.last_left_output[:]=self.left_network(self.inputs)


        self.inputs[self.input_size-2:]=self.last_left_output[:]
        self.last_right_output[:]=self.right_network(self.inputs)
        
        self.last_output=MindOutputs()

        self.last_output.set_from(self.last_left_output[3],self.last_right_output[3],
                                  np.argmax(self.last_left_output[:2]),np.argmax(self.last_right_output[:2]))

        return self.last_output
            
    def setMark(self,reward:float):

        self.left_network.evalute(reward)
        
        self.right_network.evalute(reward)