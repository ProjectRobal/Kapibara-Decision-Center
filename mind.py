
import numpy as np
from enum import Enum
from kapibara_audio import BUFFER_SIZE,SPECTOGRAM_WIDTH
from emotions import EmotionTuple

import tensorflow as tf
import time

from tensorflow.keras import layers
from tensorflow.keras import models
from transformer import TransformerBlock

import math
import os.path

from timeit import default_timer as timer
import threading
from multiprocessing import Process


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

    def __init__(self,speedA:int=0,speedB:int=0,directionA:int=0,directionB:int=0) -> None:
        self.speedA=speedA
        self.directionA=directionA

        self.speedB=speedB
        self.directionB=directionB

    def get(self)->tuple[float]:
        return (self.speedA,self.speedB,self.directionA,self.directionB)
    
    def get_norm(self)->tuple[float]:
        return (self.speedA/100.0,self.speedB/100.0,self.directionA/4.0,self.directionB/4.0)
    
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
    
class Franklin(Process):
    QUEUE_SIZE=25
    EPOCHES=50
    OUTPUT_PATH="mind.tflite"

    class Status(Enum):
        COLLECTING = 1
        ANALYZING = 2
        TRAINING = 3
        CONVERTING = 4
        DONE = 5
    
    def __init__(self,model:tf.keras.Model) -> None:
        super(Process,self).__init__()
        '''model - a keras model'''
        self.model=model
        self.frames:list[MindFrame]=[]
        self.state=self.Status.COLLECTING
        self.tflite=None
        self._run=True

    def push(self,frame:MindFrame)->None:
        if self.state!=self.Status.COLLECTING:
            return

        if len(self.frames)>=self.QUEUE_SIZE:
            self.state=self.Status.ANALYZING
            return

        self.frames.append(frame)

    def mutate(self,frame:MindFrame):
        output:MindOutputs=frame.getOutput()

        output.directionA=np.random.random()
        output.directionB=np.random.random()   
        output.speedA=np.random.random()
        output.speedB=np.random.random()  

    def analyze(self):
        '''analyze and remove/modify frames'''
        last_answer:MindFrame=self.frames[0]
        trim=False

        if last_answer.getReward()<0:
            trim=True
            self.mutate(last_answer)

        for frame in self.frames[1:]:
            if trim:
                self.frames.remove(frame)
            
            if frame.getReward()<last_answer.getReward():
                trim=True
                self.mutate(frame)
            elif frame.getReward()>last_answer.getReward():
                trim=False

            last_answer=frame

        self.state=self.Status.TRAINING

    def train(self):
        '''train with fit function'''

        inputs=([],[])
        outputs=[]

        for frame in self.frames:
            inputs[0].append(frame.getInput()[0])
            inputs[1].append(frame.getInput()[1])
            outputs.append(frame.getOutput().get_norm())

        dataset=tf.data.Dataset.from_tensor_slices(({"input_1": 
                                                     np.array(inputs[0]).reshape(len(inputs[0]),249,129,1), 
                                                     "input_2": 
                                                     np.array(inputs[1]).reshape(len(inputs[1]),1,24)}, 
                                                     np.array(outputs).reshape(len(outputs),4)))

        #lengt_dataset = dataset.reduce(0, lambda x,_: x+1).numpy()

        train_ds=dataset

        train_ds=train_ds.batch(64)

        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

        _ = self.model.fit(
            train_ds,batch_size=64,epochs=self.EPOCHES)
        
        self.state=self.Status.CONVERTING

    def convert(self):
        '''convert keras model into tf lite'''

        converter=tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter=True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
tf.lite.OpsSet.SELECT_TF_OPS]
        
        self.tflite=tf.lite.Interpreter(converter.convert())

        with open(self.OUTPUT_PATH, 'wb') as f:
            f.write(self.tflite)

        self.model.save(Mind.MIND_SAVE_PATH)

        self.tflite.allocate_tensors()

        self.state=self.Status.DONE

    def get_tf_lite(self):
        if self.state==self.Status.DONE:
            self.reset()
            return self.tflite
        else:
            return None
        
    def is_done(self)->bool:
        return self.state==self.Status.DONE
    
    def reset(self):
        self.state=self.Status.COLLECTING
        self.frames.clear()

    def _main(self):
        try:
            match self.state:
                case self.Status.COLLECTING:
                    return
                case self.Status.ANALYZING:
                    self.analyze()
                case self.Status.TRAINING:
                    self.train()
                case self.Status.CONVERTING:
                    self.convert()
                case _:
                    return
        except Exception as e:
            print(str(e))
            self.reset()
        
    def loop(self):
        while self._run:
            self._main()

    def run(self) -> None:
        self.loop()
                

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
        self.short_term_memory:list[MindFrame]=[]

    def init_memory_model(self):
        
        audio_input_layer=layers.Input(shape=(249,129,1))

        resizing=layers.Resizing(64,64)(audio_input_layer)

        # Instantiate the `tf.keras.layers.Normalization` layer.
        norm_layer = layers.Normalization()
        # Fit the state of the layer to the spectrograms
        # with `Normalization.adapt`.
        #norm_layer.adapt(data=dataset.map(map_func=lambda spec, label: spec))
        
        norm_layer(resizing)

        conv1=layers.Conv2D(64, 3, activation='relu')(resizing)

        conv2=layers.Conv2D(32, 3, activation='relu')(conv1)

        conv3=layers.Conv2D(16, 3, activation='relu')(conv2)

        maxpool=layers.MaxPooling2D()(conv3)

        dropout1=layers.Dropout(0.25)(maxpool)

        audio_output=layers.Flatten()(dropout1)

        reshape=tf.keras.layers.Reshape((1,13456))(audio_output)

        
        input=tf.keras.layers.Input(shape=(1,len(self.inputs)))

        embed=tf.keras.layers.Concatenate()([input,reshape])

        layer1=tf.keras.layers.Dense(512,activation='linear')(embed)

        layer2=tf.keras.layers.Dense(512,activation='linear')(layer1)

        layer3=tf.keras.layers.Dense(64,activation='linear')(layer2)

        output=tf.keras.layers.Dense(1,activation='linear')(layer3)

        self.long_memory=tf.keras.Model(inputs=[audio_input_layer,input],outputs=output)

        self.long_memory.compile(
            #loss=tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss"),
            loss=tf.keras.losses.MeanSquaredError(),
            #loss=tf.keras.losses.MeanAbsoluteError(),
            optimizer="adam"
        )



    
    def init_model(self):
        '''init a decision model'''
        
        
        #a root 
        audio_input_layer=layers.Input(shape=(249,129,1))

        resizing=layers.Resizing(64,64)(audio_input_layer)

        # Instantiate the `tf.keras.layers.Normalization` layer.
        norm_layer = layers.Normalization()
        # Fit the state of the layer to the spectrograms
        # with `Normalization.adapt`.
        #norm_layer.adapt(data=dataset.map(map_func=lambda spec, label: spec))
        
        norm_layer(resizing)

        conv1=layers.Conv2D(64, 3, activation='relu')(resizing)

        conv2=layers.Conv2D(32, 3, activation='relu')(conv1)

        conv3=layers.Conv2D(16, 3, activation='relu')(conv2)

        maxpool=layers.MaxPooling2D()(conv3)

        dropout1=layers.Dropout(0.25)(maxpool)

        audio_output=layers.Flatten()(dropout1)

        reshape=tf.keras.layers.Reshape((1,13456))(audio_output)

        
        input=tf.keras.layers.Input(shape=(1,len(self.inputs)))

        embed=tf.keras.layers.Concatenate()([input,reshape])

        layer2=tf.keras.layers.Dense(256,activation='relu')(embed)

        layer1=tf.keras.layers.Dense(64,activation='relu')(layer2)

        output=tf.keras.layers.Dense(4,activation='sigmoid')(layer1)

        self.model=tf.keras.Model(inputs=[audio_input_layer,input],outputs=output)

        self.model.compile(
            loss=tf.keras.losses.Huber(delta=0.1, reduction="auto", name="huber_loss"),
            #loss=tf.keras.losses.MeanSquaredError(),
            #loss=tf.keras.losses.MeanAbsoluteError(),
            optimizer="sgd"
        )

        self.model.save(self.MIND_SAVE_PATH)

        # try use tf lite model instead of normal model


    def train_test(self):

        n=16

        x=np.random.random(len(self.inputs)*n).reshape(n,1,len(self.inputs))
        spectogram=np.random.random(249*129*n).reshape(n,249,129,1)
        y=np.random.random(n*4).reshape(n,4)

        self.model.fit(x=(spectogram,x),y=y, batch_size=1024,epochs=5)

    def train_memory_test(self):

        n=100

        x=np.random.random(len(self.inputs)*n).reshape(n,1,len(self.inputs))
        spectogram=np.random.random(249*129*n).reshape(n,249,129,1)
        y=np.random.random(n).reshape(n,1)*10

        self.long_memory.fit(x=(spectogram,x),y=y, batch_size=1024,epochs=100)


    def run_model(self):

        predictions=self.model((
                                self.spectogram[tf.newaxis],
                                self.inputs.reshape(1,1,len(self.inputs)),
                                ),
                                training=False)
                
        return predictions
    
    def run_memory(self):

        return self.long_memory((
                                self.spectogram[tf.newaxis],
                                self.inputs.reshape(1,1,len(self.inputs)),
                                ),
                                training=False)
        
        

    def getData(self,data:dict):
        
        #self.gyroscope:np.array=data["Gyroscope"]["gyroscope"]/MAX_ANGEL
        #self.accelerometer:np.array=data["Gyroscope"]["acceleration"]/DISTANCE_MAX_VAL

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


    def loop(self)->MindOutputs:
        '''Work of a model:
        1. Prediction of decision model
        2. Get reward
        3. Compare it with reward returned from memory model
        4. When new reward is better than reward from memory model add it to short memory list
        5. Otherwise apply slight modification and then add it to short memory list
        6. Train decision model with one sample
        7. After N samples train decision model based on samples from short memory lists
        8. Repeat
        '''

        predictions=self.run_model()

        self.last_output=MindOutputs()

        self.last_output.set_from_norm(predictions[0][0][0],predictions[0][0][2],predictions[0][0][1],predictions[0][0][3])

        return self.last_output
    
    def setMark(self,reward:float):
        
        self.short_term_memory.append(MindFrame(inputs=self.inputs,spectogram=self.spectogram,output=self.last_output,reward=reward))