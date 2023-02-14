import network.client as client
import tensorflow as tf

import numpy as np
from scipy.signal import butter,filtfilt

from kapibara_audio import KapibaraAudio
from microphone import Microphone

import emotions

mic=Microphone(chunk=16000)

model=KapibaraAudio('./best_model')


def design_butter_lowpass_filter(cutoff,fs,order):
    normal_cutoff = (2*cutoff) / fs
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass',analog=False)

    return b,a


b,a = design_butter_lowpass_filter(1000.0,16000.0,2)



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


with client.connect('192.168.108.216:5051') as channels:
    stub=client.get_stub(channels)
    while True:

        audio=mic.record(2)/32767.0

        audio=filtfilt(b,a,audio)

        audio=tf.cast(audio,dtype=tf.float32)

        output=model.input(audio)
        #output="nervous"

        select_mood(output)

        curr_mood.loop()
        print(output)

        print("Send Command!")
        msg=client.process_data(stub,data)
        #print(msg)

