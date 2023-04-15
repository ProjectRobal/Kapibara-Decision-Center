''' A training agent for Kapibara robot 

Collect N frames from mind model,
Analyze frames and modify them it needed,
Train model with fit function,
Return it to Mind module

'''

import numpy as np
from enum import Enum
import tensorflow as tf

from mind import MindFrame,MindOutputs

QUEUE_SIZE=50

EPOCHES=50

class Franklin:
    class Status(Enum):
        COLLECTING = 1
        ANALYZING = 2
        TRAINING = 3
        CONVERTING = 4
        DONE = 5
    
    def __init__(self,model:tf.keras.Model) -> None:
        '''model - a keras model'''
        self.model=model
        self.frames:list[MindFrame]=[]
        self.state=self.Status.COLLECTING
        self.tflite=None

    def push(self,frame:MindFrame)->None:
        if self.state!=self.Status.COLLECTING:
            return

        if len(self.frames)>=QUEUE_SIZE:
            self.state=self.Status.ANALYZING
            return

        self.frames.append(frame)

    def mutate(self,frame:MindFrame):
        output:MindOutputs=frame.getOutput()

        output.directionA=np.random.random()
        output.directionB=np.random.random()        

    def analyze(self):
        '''analyze and remove/modify frames'''
        last_answer:MindFrame=self.frames[0]
        trim=False
        for frame in self.frames:
            if trim:
                self.frames.remove(frame)
            
            if frame.getReward()<last_answer.getReward():
                trim=True
                self.mutate(frame)
            elif frame.getReward()>last_answer.getReward():
                trim=False

            last_answer=frame

    def train(self):
        '''train with fit function'''

        inputs=[]
        outputs=[]

        for frame in self.frames:
            inputs.append(frame.getInput())
            outputs.append(frame.getOutput())

        dataset=tf.data.Dataset.from_tensor_slices((inputs,outputs))

        train_ds=dataset

        train_ds=train_ds.batch(64)

        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

        _ = self.model.fit(
            train_ds,
            epochs=EPOCHES
            )
        
        self.state=self.Status.CONVERTING

    def convert(self):
        '''convert keras model into tf lite'''

        converter=tf.lite.TFLiteConverter.from_keras_model(self.model)
        self.tflite=converter.convert()

        self.state=self.Status.DONE

    def get_tf_lite(self):
        if self.state==self.Status.DONE:
            self.state=self.Status.COLLECTING
            return self.tflite
        
    
    def reset(self):
        self.state=self.Status.COLLECTING
        self.frames.clear()
        

    def loop(self):
        while True:
            try:
                match self.state:
                    case self.Status.COLLECTING:
                        continue
                    case self.Status.ANALYZING:
                        self.analyze()
                    case self.Status.TRAINING:
                        self.train()
                    case self.Status.CONVERTING:
                        self.convert()
                    case _:
                        continue
            except Exception as e:
                print(e)
                self.reset()
                


