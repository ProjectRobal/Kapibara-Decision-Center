import behavior
from emotions import EmotionModifier,EmotionTuple

from modifiers import HearingCenter,FrontSensorModifier,FloorSensorModifier,ShockModifier

from mind import Mind,MindOutputs
from timeit import default_timer as timer

# emotions holder
emotions=EmotionTuple()

mind=Mind(emotions)

mind.init_model()

times=[]

for i in range(25):

    start=timer()

    mind.mutate()

    times.append(timer()-start)

    print("T: ",times[-1]," s")

avg=0
for t in times:
    avg+=t

avg=avg/25.0

print("Tavg: ",avg)
