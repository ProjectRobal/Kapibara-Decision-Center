'''

The workflow:
    get data from sensors,
    push them into EmotionsModifier and Kapibara Hearing Center,
    push data into Kapibara Decision Center
    evaluate
    update outputs

https://pygad.readthedocs.io/en/latest/README_pygad_kerasga_ReadTheDocs.html#examples

Change threshold for Shock modifier

'''

import network.client as client
from _grpc.rc_service_pb2 import _None,Motor,DistanceSensor,Gyroscope,Servo,AudioChunk,Command,Message
import numpy as np

import behavior
from emotions import EmotionModifier,EmotionTuple

from modifiers import HearingCenter,FrontSensorModifier,FloorSensorModifier,ShockModifier,BoringModifier

from mind import Mind,MindOutputs

from timeit import default_timer as timer

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
        "pwm0":45,
        "pwm1":45,
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
    ShockModifier(),
    BoringModifier()
]


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
    gyroscope:Gyroscope=Gyroscope(acceleration=[0,0,200],gyroscope=[0,0,2.0],accel_range=2**16 -1,gyro_range=2**16 -1)
    left:AudioChunk=AudioChunk(data=[0]*32000)
    right:AudioChunk=AudioChunk(data=[0]*32000)

    msg=Message(front=front,floor=floor,gyroscope=gyroscope,left=left,right=right,status=0,message="")

    return client.from_message_to_json(msg,data)

iteration:int=0

mind=Mind(emotions)

mind.init_model()


with client.connect('127.0.0.1:5051') as channels:
#if True:
    while True:
        stub=client.get_stub(channels)

        msg=client.process_data(stub,data)
        data=preprocess_data(msg,data)

        emotions.clear()

        for mod in modifiers:
            mod.retriveData(data)

        for mod in modifiers:
            mod.modify(emotions)

        data["spectogram"]=modifiers[0].get_spectogram()

        select_mood(emotions)

        mind.getData(data)

        output=mind.loop()

        data["Motors"]["speedA"]=output.motor1()[0]
        data["Motors"]["speedB"]=output.motor2()[0]
        data["Motors"]["directionA"]=output.motor1()[1]
        data["Motors"]["directionB"]=output.motor2()[1]


        # in finall version it will run in sperate thread probably
        curr_mood.loop()

        print(str(emotions))
        print("Outputs: ")
        print(str(output.get()))

        reward=emotions.estimate()

        mind.setMark(reward)

        print("Iteration: ",iteration)
        iteration+=1
