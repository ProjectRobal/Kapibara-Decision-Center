# Kapibara decision center
 A script that will controll Kapibra. It will be based on two machine learning models:
- Hearing Center - a model that will take sound from microphone and intepret it's meaning
- Decision Center - a model that will take data from all sensors and make decision based on it

# Emotions:
 Every actions and reading from sensors can be associated with emotionsm, variables that will affect a robot decision making. 
The base emotion will be:
- fear
- sadness
- anger
- pleasure
- unsettlement

Emotions can be modified by predefined scripts ( that will emulate instinctive behaviors ) or 
some machine learing models.

# Hearing Center
 A model that will take sound from on board microphones and associates specific emotion to it.
If sound will be marked as neutral it will not affect any emotions.
That model is pretrained on specific custom made dataset.

# Decision Center
 A model that will make decision ( driving motors ), based on reading from all sensors and n past actions that robot took.
The model will be trained using genetic algorithms in which fitness function will calculate fitness value using emotions.
If positive emotions dominate, the model will be taken into account. Otherwise If negative emotions dominate, the model will
be dropped.
