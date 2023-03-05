import pygad.kerasga
import pygad


import numpy as np
from kapibara_audio import BUFFER_SIZE
from emotions import EmotionTuple

import tensorflow as tf
import time

from tensorflow.keras import layers
from tensorflow.keras import models

import math
import os.path

from timeit import default_timer as timer
from tflitemodel import LiteModel

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
    

    def __init__(self,emotions:EmotionTuple,fitness,callback) -> None:
        self.last_outputs=np.array([MindOutputs(0,0,0,0)]*10)

        self.gyroscope=np.zeros(3,dtype=np.float32)
        self.accelerometer=np.zeros(3,dtype=np.float32)

        self.dis_front=0.0

        self.dis_floor=0.0

        self.audio=np.zeros(BUFFER_SIZE,dtype=np.float32)

        self.audio_coff=(0,0)

        self.emotions=emotions

        self.inputs=np.ndarray(len(self.gyroscope)+len(self.accelerometer)+len(self.audio)+
                               len(self.audio_coff)+(len(self.last_outputs)*4)+2,dtype=np.float32)
        
        #self.inputs=self.inputs.reshape(len(self.inputs),1)
        
        self.fitness=fitness
        self.callback=callback

    
    def init_model(self):

        input=layers.Input(len(self.inputs))

        layer_1=layers.Dense(512,activation="linear")(input)

        layer_2=layers.Dense(386,activation="linear")(layer_1)

        layer_out1_1=layers.Dense(256,activation="linear")(layer_2)

        layer_out1_2=layers.Dense(128,activation="sigmoid")(layer_out1_1)

        layer_out1_3=layers.Dense(64,activation="sigmoid")(layer_out1_2)


        output_1_speed=layers.Dense(1,activation="sigmoid")(layer_out1_3)
        output_1_direction=layers.Dense(1,activation="sigmoid")(layer_out1_3)


        layer_out2_1=layers.Dense(256,activation="linear")(layer_2)

        layer_out2_2=layers.Dense(128,activation="sigmoid")(layer_out2_1)

        layer_out2_3=layers.Dense(64,activation="sigmoid")(layer_out2_2)

        output_2_speed=layers.Dense(1,activation="sigmoid")(layer_out2_3)
        output_2_direction=layers.Dense(1,activation="sigmoid")(layer_out2_3)


        self.model=models.Model(inputs=input,outputs=[output_1_speed,output_1_direction,output_2_speed,output_2_direction])

        self.keras_ga=pygad.kerasga.KerasGA(model=self.model,
                                      num_solutions=10)

        initial_population=self.keras_ga.population_weights

        #print(initial_population)

        if os.path.isfile("./mind.pkl"):
            self.mind=pygad.load("./mind")
            print("Model has been loaded")
            return


        self.mind=pygad.GA(num_generations=100,
                           num_parents_mating=10,
                           initial_population=initial_population,
                           fitness_func=self.fitness,
                           on_generation=self.callback,
                           init_range_high=10,
                           init_range_low=-5,
                           parent_selection_type="rank",
                           crossover_type="scattered",
                           mutation_type="random",
                           mutation_percent_genes= 10
                           )
    

    def test_tflite(self):

        lite=LiteModel.from_keras_model(self.model)

        self.prepareInput()

        print(lite.predict(self.inputs.reshape(1,len(self.inputs))))

        for i in range(50):

            start=timer()

            print(lite.predict(self.inputs.reshape(1,len(self.inputs))))

            print(timer()-start," s")

        
    def run_model(self,solutions):

        self.prepareInput()

        predictions=pygad.kerasga.predict(model=self.model,
                        solution=solutions,
                        data=self.inputs.reshape(1,len(self.inputs)))
                
        return predictions
        
        

    def getData(self,data:dict):
        
        self.gyroscope:np.array=data["Gyroscope"]["gyroscope"]/(2**16 -1)
        self.accelerometer:np.array=data["Gyroscope"]["acceleration"]/(2**16 -1)

        self.dis_front=data["Distance_Front"]["distance"]/8160.0

        self.dis_floor=data["Distance_Floor"]["distance"]/8160.0

        left:np.array=np.array(data["Ears"]["channel1"],dtype=np.float64)/32767.0
        right:np.array=np.array(data["Ears"]["channel2"],dtype=np.float64)/32767.0

        for x in left:
            if np.isnan(x):
                print("Nan in left")
        for x in right:
            if np.isnan(x):
                print("Nan in right")

        self.audio:np.array=np.add(left,right,dtype=np.float64)/2.0

        for x in self.audio:
            if np.isnan(x):
                print("Nan in audio")

        m:float=np.mean(self.audio)

        l:float=np.mean(left)
        r:float=np.mean(right)

        #print(self.gyroscope)
        #print(self.accelerometer)
        #print(self.dis_front)
        #print(self.dis_floor)

        if m==0:
            self.audio_coff=(0.0,0.0)
            return

        self.audio_coff=(l/m,r/m)

    def prepareInput(self):
        self.inputs[0]=self.gyroscope[0]
        self.inputs[1]=self.gyroscope[1]
        self.inputs[2]=self.gyroscope[2]

        self.inputs[3]=self.accelerometer[0]
        self.inputs[4]=self.accelerometer[1]
        self.inputs[5]=self.accelerometer[2]

        self.inputs[6]=self.audio_coff[0]
        self.inputs[7]=self.audio_coff[1]

        self.inputs[8]=self.dis_front
        self.inputs[9]=self.dis_floor

        i=0

        for samp in self.audio:
            self.inputs[10+i]=samp
            i=i+1

        i=0
        l=len(self.audio)

        for out in self.last_outputs:
            put=out.get_norm()
            for x in put:
                self.inputs[10+i+l]=x
                i=i+1

    
    def push_output(self,output:MindOutputs):
        
        self.last_outputs=np.roll(self.last_outputs,-1)
        self.last_outputs[-1]=output

    def loop(self):
        
        self.mind.run()

        solution, solution_fitness, solution_idx = self.mind.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

        self.mind.save("./mind")