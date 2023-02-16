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

    def get(self)->list[float]:
        return [self.speedA,self.speedB,self.directionA,self.directionB]


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
        self.ga_instance:pygad.GA=pygad.GA(num_generations=self.NUM_GENERATIONS,
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
            put=out.get()
            for x in put:
                self.inputs[8+i+l]=x
                i=i+1
        

    def loop(self):
        
        self.ga_instance.run()

        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

        prediction = np.sum(np.array(self.inputs)*solution)
        print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))