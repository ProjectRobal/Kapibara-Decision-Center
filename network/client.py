import grpc
import _grpc.rc_service_pb2_grpc as pb2_grpc
from _grpc.rc_service_pb2 import _None,Motor,DistanceSensor,Gyroscope,Servo,AudioChunk,Command,Message
import numpy as np


def send_message(stub,speedA,directionA,speedB,directionB,angel1,angel2):

    msg=Command(mA=Motor(direction=directionA,speed=speedA),mB=Motor(direction=directionB,speed=speedB),ear1=Servo(angle=angel1),ear2=Servo(angle=angel2))

    return stub.SendCommand(msg)    

def send_message_data(stub,data):

    return send_message(stub=stub,speedA=data["Motors"]["speedA"],directionA=data["Motors"]["directionA"],speedB=data["Motors"]["speedB"],directionB=data["Motors"]["directionB"],angel1=data["Servos"]["pwm1"],angel2=data["Servos"]["pwm2"])

def process(stub,speedA,directionA,speedB,directionB,angel1,angel2):

    msg=Command(mA=Motor(direction=directionA,speed=speedA),mB=Motor(direction=directionB,speed=speedB),ear1=Servo(angle=angel1),ear2=Servo(angle=angel2))

    return stub.Process(msg)

def process_data(stub,data):

    return process(stub,speedA=data["Motors"]["speedA"],directionA=data["Motors"]["directionA"],speedB=data["Motors"]["speedB"],directionB=data["Motors"]["directionB"],angel1=data["Servos"]["pwm1"],angel2=data["Servos"]["pwm2"])


def from_message_to_json(msg:Message,input:dict={})->dict:
    output=input

    front:DistanceSensor=msg.front
    floor:DistanceSensor=msg.floor
    gyroscope:Gyroscope=msg.gyroscope
    left:AudioChunk=msg.left
    right:AudioChunk=msg.right


    output["Distance_Front"]={
        "distance":front.distance
    }
    output["Distance_Floor"]={
        "distance":front.distance
    }
    output["Gyroscope"]={
        "acceleration":np.array(gyroscope.acceleration),
        "gyroscope":np.array(gyroscope.gyroscope)
    }
    output["Ears"]={
        "channel1":np.array(left.data),
        "channel2":np.array(right.data)
    }

    return output


def get_stub(channel):
    return pb2_grpc.RCRobotStub(channel)

def connect(address):
    return grpc.insecure_channel(address)