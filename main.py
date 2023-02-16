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
from kapibara_audio import KapibaraAudio,BUFFER_SIZE
from _grpc.rc_service_pb2 import _None,Motor,DistanceSensor,Gyroscope,Servo,AudioChunk,Command,Message
import numpy as np

import behavior
from emotions import EmotionModifier,EmotionTuple

class HearingCenter(EmotionModifier):
    def __init__(self) -> None:
        self.hearing=KapibaraAudio('./hearing')
        self.audio=np.zeros(BUFFER_SIZE,np.int16)
        # a vector of x/m where x is mean value of channel and m is mean value of added signals
        #self.coefficient=(0,0)
    
    def retriveData(self,data:dict):
        try:
            '''get a specific data from host'''
            left:np.array=np.array(data["Ears"]["channel1"],dtype=np.float32)/32767.0
            right:np.array=np.array(data["Ears"]["channel2"],dtype=np.float32)/32767.0

            self.audio:np.array=np.add(left,right,dtype=np.float32)/2

            #m:float=np.mean(self.audio,dtype=np.float32)
            #l:float=np.mean(left,dtype=np.float32)
            #r:float=np.mean(right,dtype=np.float32)

            #self.coefficient=(l/m,r/m)

        except:
            print("Audio data is missing!")


    def modify(self,emotions:EmotionTuple):
    
        output=self.hearing.input(self.audio)

        if output=="unsettling":
            emotions.unsettlement=1
        elif output=="pleasent":
            emotions.pleasure=1
        elif output=="scary":
            emotions.fear=1
        elif output=="nervous":
            emotions.anger=1


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
curr_mood:behavior.Emotion=moods["neutral"]

# emotions holder
emotions=EmotionTuple()

modifiers:list[EmotionModifier]=[]



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


with client.connect('192.168.108.216:5051') as channels:
    stub=client.get_stub(channels)
    
    while True:
        msg=client.send_message_data(stub,data)

        data=preprocess_data(msg)

        for mod in modifiers:
            mod.retriveData(data)

        for mod in modifiers:
            mod.modify(emotions)

        #audio=mic.record(2)/32767.0

        #audio=filtfilt(b,a,audio)

        #audio=tf.cast(audio,dtype=tf.float32)

        #output="nervous"

        #select_mood(output)

        curr_mood.loop()
        #print(output)

        print("Send Command!")
        msg=client.send_message_data(stub,data)
        #print(msg)