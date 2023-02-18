import pygad.kerasga
import pygad


import numpy as np
from kapibara_audio import BUFFER_SIZE
from emotions import EmotionTuple

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

class MindOutputs:
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

        return error

    def get(self)->list[float]:
        return [self.speedA,self.speedB,self.directionA,self.directionB]
    
    def get_norm(self)->list[float]:
        return [self.speedA/100.0,self.speedB/100.0,self.directionA/3.0,self.directionB/3.0]
    
    def set_from_norm(self,speedA,speedB,directionA,directionB):
        self.speedA=speedA*100
        self.speedB=speedB*100
        self.directionA=directionA*3
        self.directionB=directionB*3

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
    

    def __init__(self,emotions:EmotionTuple,fitness) -> None:
        self.last_outputs=np.array([MindOutputs(0,0,0,0)]*10)

        self.gyroscope=np.zeros(3,dtype=np.float32)
        self.accelerometer=np.zeros(3,dtype=np.float32)

        self.audio=np.zeros(BUFFER_SIZE,dtype=np.float32)

        self.audio_coff=(0,0)

        self.emotions=emotions

        self.inputs=np.ndarray(len(self.gyroscope)+len(self.accelerometer)+len(self.audio)+
                               len(self.audio_coff)+(len(self.last_outputs)*4),dtype=np.float32)
        
        #self.inputs=self.inputs.reshape(len(self.inputs),1)
        
        self.fitness=fitness
    

    def init_model(self):

        input=layers.Input(len(self.inputs))

        layer_1=layers.Dense(512,activation="linear")(input)

        layer_2=layers.Dense(386,activation="linear")(layer_1)

        layer_out1_1=layers.Dense(256,activation="linear")(layer_2)

        layer_out1_2=layers.Dense(128,activation="linear")(layer_out1_1)

        layer_out1_3=layers.Dense(64,activation="linear")(layer_out1_2)


        output_1_speed=layers.Dense(1,activation="relu")(layer_out1_3)
        output_1_direction=layers.Dense(1,activation="relu")(layer_out1_3)


        layer_out2_1=layers.Dense(256,activation="linear")(layer_2)

        layer_out2_2=layers.Dense(128,activation="linear")(layer_out2_1)

        layer_out2_3=layers.Dense(64,activation="linear")(layer_out2_2)

        output_2_speed=layers.Dense(1,activation="relu")(layer_out2_3)
        output_2_direction=layers.Dense(1,activation="relu")(layer_out2_3)


        self.model=models.Model(inputs=input,outputs=[output_1_speed,output_1_direction,output_2_speed,output_2_direction])

        self.keras_ga=pygad.kerasga.KerasGA(model=self.model,
                                      num_solutions=10)

        initial_population=self.keras_ga.population_weights

        #print(initial_population)

        self.mind=pygad.GA(num_generations=10,
                           num_parents_mating=5,
                           initial_population=initial_population,
                           fitness_func=self.fitness)
        
    def run_model(self,solutions):

        self.prepareInput()

        predictions=pygad.kerasga.predict(model=self.model,
                        solution=solutions,
                        data=self.inputs.reshape(1,len(self.inputs)))
        

                
        return predictions
        
        

    def getData(self,data:dict):
        
        self.gyroscope:np.array=data["Gyroscope"]["gyroscope"]/(2**16 -1)
        self.accelerometer:np.array=data["Gyroscope"]["acceleration"]/(2**16 -1)

        left:np.array=np.array(data["Ears"]["channel1"],dtype=np.float32)/32767.0
        right:np.array=np.array(data["Ears"]["channel2"],dtype=np.float32)/32767.0

        self.audio:np.array=np.add(left,right,dtype=np.float32)/2

        m:float=np.mean(self.audio)

        l:float=np.mean(left)
        r:float=np.mean(right)

        if m==0:
            self.audio_coff=(0,0)
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

        i=0

        for samp in self.audio:
            self.inputs[8+i]=samp 
            i=i+1

        i=0
        l=len(self.audio)

        for out in self.last_outputs:
            put=out.get_norm()
            for x in put:
                self.inputs[8+i+l]=x
                i=i+1

    
    def push_output(self,output:MindOutputs):
        
        self.last_outputs=np.roll(self.last_outputs,-1)
        self.last_outputs[-1]=output

    def loop(self):
        
        self.mind.run()

        solution, solution_fitness, solution_idx = self.mind.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

        prediction = np.sum(np.array(self.inputs)*solution)
        print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

        self.mind.save("./mind.kdm")