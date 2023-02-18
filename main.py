'''

The workflow:
    get data from sensors,
    push them into EmotionsModifier and Kapibara Hearing Center,
    push data into Kapibara Decision Center
    evaluate
    update outputs

https://pygad.readthedocs.io/en/latest/README_pygad_kerasga_ReadTheDocs.html#examples

I am going to use EasyGA instead

'''

import network.client as client
from _grpc.rc_service_pb2 import _None,Motor,DistanceSensor,Gyroscope,Servo,AudioChunk,Command,Message
import numpy as np

import behavior
from emotions import EmotionModifier,EmotionTuple

from modifiers import HearingCenter,FrontSensorModifier,FloorSensorModifier,ShockModifier

from mind import Mind,MindOutputs

data:dict = {
    "Motors":
    {
        "speedA":0,
        "directionA":1,
        "speedB":0,
        "directionB":1
    },
    "Servos":
    {
        "pwm1":45,
        "pwm2":45,
    }
}

moods:list=[
    behavior.Neutral(data["Servos"]),
    behavior.Unsettlment(data["Servos"]),
    behavior.Pleasure(data["Servos"]),
    behavior.Fear(data["Servos"]),
    behavior.Anger(data["Servos"])
]

# curr machine mood used to drive servo ears
curr_mood:behavior.Emotion=moods[0]

# emotions holder
emotions=EmotionTuple()

modifiers:list[EmotionModifier]=[
    HearingCenter(),
    FrontSensorModifier("Distance_Front"),
    FloorSensorModifier("Distance_Floor"),
    ShockModifier()
]


#exit()

def select_mood(emotions:EmotionTuple):
    global curr_mood
    list:list[float]=emotions.get_list()

    # neutral if no emotions is dominating
    if np.var(list) < 0.02:
        curr_mood=moods[0]

    index:int=np.argmax(list)

    curr_mood=moods[index+1]
    #curr_mood=moods[output]


def preprocess_data(msg:Message,data:dict):
    '''convert grpc message to json'''
    return client.from_message_to_json(msg,data)


def placeholder_data(data:dict):

    front:DistanceSensor=DistanceSensor(distance=500)
    floor:DistanceSensor=DistanceSensor(distance=5)
    gyroscope:Gyroscope=Gyroscope(acceleration=[0,0,0],gyroscope=[0,0,0],accel_range=[2**16 -1],gyro_range=[2**16 -1])
    left:AudioChunk=AudioChunk(data=[0]*32000)
    right:AudioChunk=AudioChunk(data=[0]*32000)

    msg=Message(front=front,floor=floor,gyroscope=gyroscope,left=left,right=right,status=0,message="")

    return client.from_message_to_json(msg,data)

with client.connect('192.168.108.216:5051') as channels:
#if True:
    stub=client.get_stub(channels)

    def fitness_func(solution, solution_idx):
        global mind,data
        msg=client.send_message_data(stub,data)
        #data=placeholder_data(data)

        emotions.clear()

        for mod in modifiers:
            mod.retriveData(data)

        for mod in modifiers:
            mod.modify(emotions)

        select_mood(emotions)

        mind.getData(data)
        
        predictions=mind.run_model(solution)

        print("Output: ")
        print(predictions)

        output=MindOutputs()

        output.set_from_norm(predictions[0],predictions[2],predictions[1],predictions[3])

        err=output.error()

        mind.push_output(output)

        return emotions.estimate()+err
    
    mind=Mind(emotions,fitness_func)

    mind.init_model()
    
    mind.loop()