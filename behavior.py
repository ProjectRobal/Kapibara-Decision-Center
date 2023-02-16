import time


class Emotion:
    '''servos - a reference to servos channels'''
    def __init__(self,servos):
        self._servos = servos

    def servo1(self,pwm=None):
        if pwm is None:
            return self._servos["pwm1"]

        self._servos["pwm1"] = pwm

    def servo2(self,pwm=None):
        if pwm is None:
            return self._servos["pwm2"]

        self._servos["pwm2"] = pwm

    def loop(self):
        pass


class Neutral(Emotion):
    def __init__(self,servos):
        super().__init__(servos)

    def loop(self):
        self.servo1(90)
        self.servo2(90)


class Unsettlment(Emotion):
    def __init__(self,servos):
        super().__init__(servos)

    def loop(self):
        self.servo1(105)
        self.servo2(105)

class Pleasure(Emotion):
    def __init__(self,servos):
        super().__init__(servos)
        self.last=time.time_ns()/1000
        self.shrung=False

    def loop(self):
        if self.shrung:
            self.servo1(180)
            self.servo2(180)
        else:
            self.servo1(90)
            self.servo2(90)

        if time.time_ns()/1000-self.last >= 100:
            self.shrung=not self.shrung
            self.last=time.time_ns()/1000

class Fear(Emotion):
    def __init__(self,servos):
        super().__init__(servos)
        self.last=time.time_ns()/1000
        self.shrung=False

    def loop(self):

        if self.shrung:
            self.servo1(10)
            self.servo2(10)
        else:
            self.servo1(5)
            self.servo2(5)

        if time.time_ns()/1000-self.last >= 100:
            self.shrung=not self.shrung
            self.last=time.time_ns()/1000

class Anger(Emotion):
    def __init__(self,servos):
        super().__init__(servos)

    def loop(self):
        self.servo1(135)
        self.servo2(135)