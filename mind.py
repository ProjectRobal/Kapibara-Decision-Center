import pygad

import numpy as np
from kapibara_audio import BUFFER_SIZE
from emotions import EmotionTuple


class MindOutputs:
    def __init__(self,speedA=0,speedB=0,directionA=0,directionB=0) -> None:
        self.speedA=speedA
        self.directionA=directionA

        self.speedB=speedB
        self.directionB=directionB


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
    

    def __init__(self,emotions:EmotionTuple,fitness:function) -> None:
        self.last_outputs=np.array([MindOutputs]*10,dtype=MindOutputs)

        self.gyroscope=np.zeros(3,dtype=np.float32)
        self.accelerometer=np.zeros(3,dtype=np.float32)

        self.audio=np.zeros(BUFFER_SIZE,dtype=np.float32)

        self.audio_coff=(0,0)

        self.emotions=emotions

        self.inputs=np.ndarray(len(self.gyroscope)+len(self.accelerometer)+len(self.audio)+
                               len(self.audio_coff)+len(self.last_outputs),dtype=np.float32)
        
        self.fitness=fitness


    def init_model(self):
        self.ga_instance=pygad.GA(num_generations=self.NUM_GENERATIONS,
                       num_parents_mating=10,
                       fitness_func=self.fitness,
                       sol_per_pop=8,
                       num_genes=len(self.inputs),
                       init_range_low=-5,
                       init_range_high=10,
                       parent_selection_type="sss",
                       keep_parents=4,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=10)

    def getData(self,data:dict):
        
        self.gyroscope:np.array=data["Gyroscope"]["gyroscope"]/(2**16 -1)
        self.accelerometer:np.array=data["Gyroscope"]["acceleration"]/(2**16 -1)

        left:np.array=np.array(data["Ears"]["channel1"],dtype=np.float32)/32767.0
        right:np.array=np.array(data["Ears"]["channel2"],dtype=np.float32)/32767.0

        self.audio:np.array=np.add(left,right,dtype=np.float32)/2

        m=np.mean(self.audio)

        l=np.mean(left)
        r=np.mean(right)

        self.audio_coff=(l/m,r/m)

    def loop(self):
        pass