from numba import njit
from numba import int32, float32    # import the types
from numba.experimental import jitclass

@jitclass([
    ("fear",float32),
    ("anger",float32),
    ("pleasure",float32),
    ("unsettlement",float32)
])
class EmotionTuple:
    '''A class that will hold all emotion coefficients,
    each variable has range <0,1>'''
    def __init__(self) -> None:
        self.fear=0.0
        self.anger=0.0
        self.pleasure=0.0
        self.unsettlement=0.0

    def get_list(self) ->list[float]:
        return [self.unsettlement,self.pleasure,self.fear,self.anger]
    
    def estimate(self)->float:
        '''A function that will be used in genetic algorithm for flatten function'''
        estimation=(self.pleasure*10)-(self.fear*5)-(self.anger*2)-(self.unsettlement*1)
        
        return estimation
    
    def clear(self):
        self.fear=0.0
        self.anger=0.0
        self.pleasure=0.0
        self.unsettlement=0.0

    def __str__(self) -> str:
        return "fear: "+str(self.fear)+"\n"+"anger: "+str(self.anger)+"\n"+"pleasure: "+str(self.pleasure)+"\n""unsettlement: "+str(self.unsettlement)+"\n"


class EmotionModifier:
    '''A base class for emotion modification base on specific sensor'''
    def __init__(self) -> None:
        pass

    def retriveData(self,data:dict):
        '''get a specific data from host'''
        pass

    def modify(self,emotions:EmotionTuple):
        raise NotImplementedError()
