import behavior
from emotions import EmotionModifier,EmotionTuple

from modifiers import HearingCenter,FrontSensorModifier,FloorSensorModifier,ShockModifier

from mind import Mind,MindOutputs
from timeit import default_timer as timer

# emotions holder
emotions=EmotionTuple()

mind=Mind(emotions)

mind.init_model()

for i in range(25):

    start=timer()

    mind.mutate()

    print("T: ",timer()-start," s")
