
import numpy as np
from enum import Enum
from kapibara_audio import BUFFER_SIZE,SPECTOGRAM_WIDTH
from emotions import EmotionTuple

import tensorflow as tf
import time

from KCN.network import Network
from KCN.layer import RecurrentLayer,Layer
from KCN.activation.sigmoid import Sigmoid
from KCN.activation.linear import Linear
from KCN.activation.relu import Relu
from KCN.initializer.gaussinit import GaussInit
from KCN.BreedStrategy import BreedStrategy

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

MAX_SPEED=40

ACCELERATION_MAX=(2.0/32767.0) # in mm
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

    def get(self)->tuple[np.float64]:
        return (self.speedA,self.speedB,self.directionA,self.directionB)
    
    def get_norm(self)->tuple[np.float64]:
        return (self.speedA/100.0,self.speedB/100.0,self.directionA/4.0,self.directionB/4.0)
    
    def get_reward(self)->np.float64:
        return self.reward
    
    def set_reward(self,reward:np.float64):
        self.reward=reward
    
    def set_from(self,speedA:np.float64,speedB:np.float64,directionA:np.float64,directionB:np.float64):

        self.speedA=(speedA*100)
        self.speedB=(speedB*100)

        if self.speedA>MAX_SPEED:
            self.speedA=MAX_SPEED
        if self.speedB>MAX_SPEED:
            self.speedB=MAX_SPEED

        if directionA<3:
            self.directionA=directionA
        else:
            self.directionA=3
        
        if directionB<3:
            self.directionB=directionB
        else:
            self.directionB=3

    def motor1(self)->tuple[int,int]:
        return (int(self.speedA),int(self.directionA))
    
    def motor2(self)->tuple[int,int]:
        return (int(self.speedB),int(self.directionB))
    
class MindFrame:
    '''A pair of inputs and outputs of mind'''
    def __init__(self,inputs,spectogram,output:MindOutputs,reward:np.float64) -> None:
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

        self.last_eval:float=0.0
        self.last_rewad:float=0.0

        self.gyroscope=np.zeros(3,dtype=np.float32)
        self.accelerometer=np.zeros(3,dtype=np.float32)

        self.audio=np.zeros(BUFFER_SIZE,dtype=np.float32)

        self.front_sensors=np.zeros(FRONT_SENSORS_COUNT,dtype=np.float32)

        self.floor_sensors=np.zeros(FLOOR_SENSORS_COUNT,dtype=np.float32)

        self.audio_coff=(0,0)

        self.emotions=emotions

        ''' 0...2 gyroscope, 3...5 accelerometer, 6...7 audio coefficient, floor_sensors, 
        front_sensors, spectogram, 2 outputs from second network'''

        self.input_size=len(self.gyroscope)+len(self.accelerometer)+len(self.audio_coff)+FLOOR_SENSORS_COUNT+FRONT_SENSORS_COUNT+4+(32*32)+1

        self.initializer=GaussInit(0,0.0001)
        self.initializer1=GaussInit(0,0.5)

        self.inputs=np.ndarray(self.input_size,dtype=np.float64)
        self.last_left_output=np.zeros(4,dtype=np.float64)
        self.last_right_output=np.zeros(4,dtype=np.float64)

        self.left_network=Network(self.input_size)
        self.left_network.last_eval=0
        self.left_network.setTrendFunction(self.TrendFunction)

        self.left_network.addLayer(256,16,RecurrentLayer,[Relu]*256,self.initializer)

        self.left_network_motors=Network(256)
        self.left_network_motors.last_eval=0
        self.left_network_motors.setTrendFunction(self.TrendFunction)

        self.left_network_motors.addLayer(1,4,Layer,[Relu],self.initializer1)

        self.left_network_dir=Network(256)
        self.left_network_dir.last_eval=0
        self.left_network_dir.setTrendFunction(self.TrendFunction)

        self.left_network_dir.addLayer(3,4,Layer,[Relu,Relu,Relu],self.initializer)

        self.right_network=Network(self.input_size)
        self.right_network.last_eval=0
        self.right_network.setTrendFunction(self.TrendFunction)

        self.right_network.addLayer(256,16,RecurrentLayer,[Relu]*256,self.initializer)

        self.right_network_motors=Network(256)
        self.right_network_motors.last_eval=0
        self.right_network_motors.setTrendFunction(self.TrendFunction)

        self.right_network_motors.addLayer(1,4,Layer,[Relu],self.initializer1)

        self.right_network_dir=Network(256)
        self.right_network_dir.last_eval=0
        self.right_network_dir.setTrendFunction(self.TrendFunction)

        self.right_network_dir.addLayer(3,4,Layer,[Relu,Relu,Relu],self.initializer)
        
        self.spectogram=None

    def getData(self,data:dict):

        self.inputs[0:3]=data["Gyroscope"]["gyroscope"]/MAX_ANGEL
        self.inputs[4:7]=data["Gyroscope"]["acceleration"]/ACCELERATION_MAX

        self.inputs[8]=data["Distance_Front"]["distance"]/8160.0
        self.inputs[9]=data["Distance_Front1"]["distance"]/8160.0

        self.inputs[10]=data["Distance_Floor"]["distance"]/8160.0

        left:np.array=np.array(data["Ears"]["channel1"],dtype=np.float32)/32767.0
        right:np.array=np.array(data["Ears"]["channel2"],dtype=np.float32)/32767.0

        self.audio:np.array=np.add(left,right,dtype=np.float32)/2.0

        print("Audio ",self.audio)

        m:float=np.max(self.audio)

        l:float=np.max(left)
        r:float=np.max(right)

        if m==0.0:
            self.audio_coff=(0.0,0.0)
        else:
            self.audio_coff=(l/m,r/m)

        self.inputs[11]=self.audio_coff[0]
        self.inputs[12]=self.audio_coff[1]

        self.spectogram=data["spectogram"]

        print(cv2.resize(self.spectogram.numpy(),(32,32)).reshape(32*32,))

        self.inputs[13:-3]=cv2.resize(self.spectogram.numpy(),(32,32)).reshape(32*32,)

        print("Inputs: ",self.inputs[-5:])
        #input()

    def eval_filter(self,deval:float):
        return np.clip(np.exp(4*deval-12)-np.exp(-4*deval-12),-0.3,1.0)

    def TrendFunction(self,eval:float,network:Network)->float:

        out:float=self.eval_filter(eval-network.last_eval)

        network.last_eval=eval

        print("dEval: ",out)

        if eval>=0.0:
            return 1.0

        return out
        
    def loop(self)->MindOutputs:


        self.inputs[self.input_size-4:]=self.last_right_output[:]
        
        first_outputs=self.left_network.step(self.inputs)

        self.last_left_output[3]=self.left_network_motors.step(first_outputs)[0]
        self.last_left_output[:3]=self.left_network_dir.step(first_outputs)[:]


        self.inputs[self.input_size-4:]=self.last_left_output[:]
        
        first_outputs=self.right_network.step(self.inputs)

        self.last_right_output[3]=self.right_network_motors.step(first_outputs)[0]
        self.last_right_output[:3]=self.right_network_dir.step(first_outputs)[:]

        print("Right output: ",self.last_right_output)
        print("Left output: ",self.last_left_output)
        #input()
        
        self.last_output=MindOutputs()

        self.last_output.set_from(self.last_left_output[3],self.last_right_output[3],
                                  np.argmax(self.last_left_output[:3]),np.argmax(self.last_right_output[:3]))
        
        return self.last_output
            
    def setMark(self,reward:float):

        R:float=reward

        #if reward>0:
        #    R=200
        #else:
        #    R=1000.0*(reward-self.last_rewad)


        self.left_network.evalute(R)
        self.left_network_dir.evalute(R)
        self.left_network_motors.evalute(R)

        self.right_network.evalute(R)
        self.right_network_dir.evalute(R)
        self.right_network_motors.evalute(R)

        self.last_rewad=reward