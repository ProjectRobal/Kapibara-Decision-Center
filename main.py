'''

The workflow:
    get data from sensors,
    push them into EmotionsModifier and Kapibara Hearing Center,
    push data into Kapibara Decision Center
    evaluate
    update outputs

https://pygad.readthedocs.io/en/latest/README_pygad_kerasga_ReadTheDocs.html#examples
'''

import network.client as client
from kapibara_audio import KapibaraAudio
from _grpc.rc_service_pb2 import _None,Motor,DistanceSensor,Gyroscope,Servo,AudioChunk,Command,Message

import emotions

class EmotionTuple:
    '''A class that will hold all emotion coefficients,
    each variable has range <0,1>'''
    def __init__(self) -> None:
        self.fear=0.0
        self.sadness=0.0
        self.pleasure=0.0
        self.unsettlement=0.0
        self.last_estimation=0.0
    
    def estimate(self)->float:
        '''A function that will be used in genetic algorithm for flatten function'''
        estimation=(self.pleasure*10)-(self.fear*5)-(self.sadness*2)-(self.unsettlement*1)
        
        return estimation


class EmotionModifier:
    '''A base class for emotion modification base on specific sensor'''
    def __init__(self) -> None:
        pass

    def retriveData(self,data):
        '''get a specific data from host'''
        pass

    def modify(self,emotions:EmotionTuple):
        raise NotImplementedError()


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

moods:dict={
    "neutral":emotions.Neutral(data["Servos"]),
    "unsettling":emotions.Unsettlment(data["Servos"]),
    "pleasent":emotions.Pleasure(data["Servos"]),
    "scary":emotions.Fear(data["Servos"]),
    "nervous":emotions.Anger(data["Servos"])
}

curr_mood:emotions.Emotion=moods["neutral"]

def select_mood(output):
    global curr_mood
    curr_mood=moods[output]


hearing=KapibaraAudio('./hearing')



def preprocess_data(msg:Message,data:dict):
    '''convert grpc message to json'''
    return client.from_message_to_json(msg,data)



with client.connect('192.168.108.216:5051') as channels:
    stub=client.get_stub(channels)
    
    while True:
        msg=client.send_message(stub,data)

        data=preprocess_data(data)
        #audio=mic.record(2)/32767.0

        #audio=filtfilt(b,a,audio)

        #audio=tf.cast(audio,dtype=tf.float32)

        output=hearing.input(msg.left)
        #output="nervous"

        select_mood(output)

        curr_mood.loop()
        print(output)

        print("Send Command!")
        msg=client.send_message_data(stub,data)
        #print(msg)