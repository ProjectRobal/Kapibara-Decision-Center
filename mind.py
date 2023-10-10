
import numpy as np
from enum import Enum
from kapibara_audio import BUFFER_SIZE,SPECTOGRAM_WIDTH
from emotions import EmotionTuple

import tensorflow as tf
import time

from tensorflow.keras import layers
from tensorflow.keras import models


import math
import os.path

from timeit import default_timer as timer
import threading
from multiprocessing import Process
import copy


MODEL_PATH='mind.tf'

'''
Gyroscope is now change of rotation,
Accelration is now a change of position

'''

DISTANCE_MAX_VAL=2048.0 # in mm
MAX_ANGEL=360.0

FLOOR_SENSORS_COUNT=8
FRONT_SENSORS_COUNT=8

class MindOutputs:

    MAX_INPUT_VALUE=4294967295.0

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
    
    def set_from_norm(self,speedA:float,speedB:float,directionA:float,directionB:float):

        self.speedA=speedA*100.0
        self.speedB=speedB*100.0

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
                

SHORT_MEMORY_CAPACITY=200

class MindDataset:
    def __init__(self):
        self.spectograms=np.zeros((SHORT_MEMORY_CAPACITY,249,129,1))
        self.inputs=np.zeros((SHORT_MEMORY_CAPACITY,1,24))
        self.outputs=np.zeros((SHORT_MEMORY_CAPACITY,4))
        self.rewards=np.zeros(SHORT_MEMORY_CAPACITY)
        self.i=0

    def push(self,input,spectogram,output,reward):
        self.i=self.i%SHORT_MEMORY_CAPACITY

        self.inputs[self.i]=input.reshape((1,24))
        self.spectograms[self.i]=spectogram
        self.outputs[self.i]=np.array(output)*100.0
        self.rewards[self.i]=reward

        self.i=self.i+1


    def clear(self):
        self.spectograms=np.zeros((SHORT_MEMORY_CAPACITY,249,129,1))
        self.inputs=np.zeros((SHORT_MEMORY_CAPACITY,1,24))
        self.outputs=np.zeros((SHORT_MEMORY_CAPACITY,4))
        self.rewards=np.zeros(SHORT_MEMORY_CAPACITY)
        self.i=0

    def full(self):
        return self.i==SHORT_MEMORY_CAPACITY

    def len(self):
        return self.i
    
    def y(self):
        return (self.outputs,self.rewards)
    
    def x(self):
        return (self.spectograms,self.inputs)


class Mind:
    MIND_SAVE_PATH="model.tf"

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


        self.last_outputs=np.array([MindOutputs(0,0,0,0)]*10)

        self.gyroscope=np.zeros(3,dtype=np.float32)
        self.accelerometer=np.zeros(3,dtype=np.float32)

        self.audio=np.zeros(BUFFER_SIZE,dtype=np.float32)

        self.front_sensors=np.zeros(FRONT_SENSORS_COUNT,dtype=np.float32)

        self.floor_sensors=np.zeros(FLOOR_SENSORS_COUNT,dtype=np.float32)

        self.audio_coff=(0,0)

        self.emotions=emotions

        ''' 0...2 gyroscope, 3...5 accelerometer, 6...7 audio coefficient, floor_sensors, 
        front_sensors, spectogram'''

        self.inputs=np.ndarray(len(self.gyroscope)+len(self.accelerometer)+
                               len(self.audio_coff)+FLOOR_SENSORS_COUNT+FRONT_SENSORS_COUNT,dtype=np.float32)
        
        self.spectogram=None

        self.long_memory=None

        # a dictionary that stores pairs of ( (input,output), reward )
        self.short_term_memory=MindDataset()

    
    def init_model(self):
        '''init a decision model'''

        if os.path.exists(MODEL_PATH):
            self.model=tf.keras.models.load_model(MODEL_PATH)
        
        
        #a root 
        audio_input_layer=layers.Input(shape=(249,129,1))

        resizing=layers.Resizing(32,32)(audio_input_layer)

        # Instantiate the `tf.keras.layers.Normalization` layer.
        norm_layer = layers.Normalization()
        # Fit the state of the layer to the spectrograms
        # with `Normalization.adapt`.
        #norm_layer.adapt(data=dataset.map(map_func=lambda spec, label: spec))
        
        norm_layer(resizing)

        conv1=layers.Conv2D(64, 3, activation='linear')(resizing)

        conv2=layers.Conv2D(32, 3, activation='linear')(conv1)

        conv3=layers.Conv2D(16, 3, activation='linear')(conv2)

        maxpool=layers.MaxPooling2D()(conv3)

        dropout1=layers.Dropout(0.25)(maxpool)

        audio_output=layers.Flatten()(dropout1)

        reshape=tf.keras.layers.Reshape((1,2704))(audio_output)

        
        input=tf.keras.layers.Input(shape=(1,len(self.inputs)))

        embed=tf.keras.layers.Concatenate()([input,reshape])

        layer2=tf.keras.layers.Dense(256,activation='linear')(embed)

        layer1=tf.keras.layers.Dense(64,activation='linear')(layer2)

        output=tf.keras.layers.Dense(4,activation='relu',name="output")(layer1)

        reward=tf.keras.layers.Dense(1,activation="linear",name="reward")(layer1)

        self.model=tf.keras.Model(inputs=[audio_input_layer,input],outputs=[output,reward])


        self.model.compile(
            loss=tf.keras.losses.Huber(delta=0.9, reduction="auto", name="huber_loss"),
            #loss=tf.keras.losses.MeanSquaredError(),
            #loss=tf.keras.losses.MeanAbsoluteError(),s
            optimizer="adam"
        )

        self.model.save(MODEL_PATH)


    def train_test(self):

        n=200

        x=np.random.random(len(self.inputs)*n).reshape(n,1,len(self.inputs))
        spectogram=np.random.random(249*129*n).reshape(n,249,129,1)
        y=np.random.random(n*4).reshape(n,4)
        r=np.random.random(n).reshape(n,1)*10

        self.model.fit(x=(spectogram,x),y=(y,r), batch_size=16,epochs=20)

    def run_model(self):

        predictions=self.model((
                                self.spectogram[tf.newaxis],
                                self.inputs.reshape(1,1,len(self.inputs)),
                                ),
                                training=False)
                
        return predictions
        
        

    def getData(self,data:dict):

        self.inputs[0:3]=data["Gyroscope"]["gyroscope"]/MAX_ANGEL
        self.inputs[4:7]=data["Gyroscope"]["acceleration"]/DISTANCE_MAX_VAL

        self.inputs[10]=data["Distance_Front"]["distance"]/8160.0

        self.inputs[18]=data["Distance_Floor"]["distance"]/8160.0

        left:np.array=np.array(data["Ears"]["channel1"],dtype=np.float32)/32767.0
        right:np.array=np.array(data["Ears"]["channel2"],dtype=np.float32)/32767.0

        self.audio:np.array=np.add(left,right,dtype=np.float32)/2.0

        m:float=np.mean(self.audio)

        l:float=np.mean(left)
        r:float=np.mean(right)

        if m==0:
            self.audio_coff=(0.0,0.0)
        else:
            self.audio_coff=(l/m,r/m)

        self.inputs[8]=self.audio_coff[0]
        self.inputs[9]=self.audio_coff[1]

        self.spectogram=data["spectogram"]


    def process_train(self,model,x,y):

        model.fit(x=x,y=y, batch_size=16,epochs=20,verbose=0)

        model.save(MODEL_PATH)

    def memorize(self):
        '''
        A function that do memorizing step
        '''
        if not self.short_term_memory.full():
            return

        x=copy.deepcopy(self.short_term_memory.x())

        y=copy.deepcopy(self.short_term_memory.y())

        self.short_term_memory.clear()

        #train_model=tf.keras.models.clone_model(self.model)

        #trainer=Process(target=self.process_train,args=(self.model,x,y,))

        #trainer.start()

        self.process_train(self.model,x,y)

        #trainer.join()

        
    def loop(self)->MindOutputs:
        '''Work of a model:
        1. Prediction of decision model
        2. Get reward
        3. Compare it with reward returned from model
        4. When new reward is better than reward from model, add it to short memory list
        5. Otherwise apply slight modification and then add it to short memory list
        6. After N samples train decision model based on samples from short memory lists
        7. Repeat

        * Run trainings in seaperate networks
        '''

        predictions=self.run_model()

        self.last_output=MindOutputs()

        outputs=((np.array(predictions[0])/100.0)%1).reshape(4)

        self.last_output.set_from_norm(outputs[0],
                                       outputs[2]
                                       ,outputs[1]
                                       ,outputs[3])

        self.last_output.set_reward(predictions[1][0][0][0])

        return self.last_output
            
    def setMark(self,reward):

        if reward < self.last_output.reward or reward < 0:
            self.last_output.set_from_norm(np.random.random(),np.random.random(),np.random.random(),np.random.random())

        self.short_term_memory.push(self.inputs,self.spectogram,self.last_output.get_norm(),reward)