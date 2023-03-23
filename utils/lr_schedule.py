import math

def linear(y1=0.0, y2=1.0, steps=100):
    return lambda x: (1 - x / (steps - 1)) * (y1- y2) + y2

def step(y1=0.0, y2=1.0, steps=100):
    level = 5
    per_steps =  steps//level
    return lambda x: y1  +  (y2- y1)* (x//per_steps / level)

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def two_cycle(y1=0.0, y2=1.0, steps=100):
    # cos(0, pi) + cos (0, pi)
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    def func(x):
        rush_down_epoch = 300
        if x <= rush_down_epoch:
            lr_e = ((1 - math.cos(x * math.pi / rush_down_epoch)) / 2) * (y1/10 - y1) + y1
        else:
            lr_e = ((1 - math.cos((x-rush_down_epoch) * math.pi / (steps-rush_down_epoch))) / 2) * (y2 - y1/10) + y1/10
        return lr_e
    return func
def soft_two_cycle(y1=0.0, y2=1.0, steps=100):
    # cos(0, pi) + cos (2/pi, pi)
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    def func(x):
        rush_down_epoch = 300
        if x <= rush_down_epoch:
            lr_e = ((1 - math.cos(x * math.pi / rush_down_epoch)) / 2) * (y1/10 - y1) + y1
        else:
            lr_e = (- math.cos((x-rush_down_epoch) * math.pi / (steps-rush_down_epoch)/2 + math.pi/2)) * (y2 - y1/10) + y1/10
        return lr_e
    return func

def more_soft_two_cycle(y1=0.0, y2=1.0, steps=100):
    # 1/2 epoch * cos(0, pi) + cos (2/pi, pi)
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    def func(x):
        rush_down_epoch = 300
        half_rush_e = round(rush_down_epoch/2)
        if x < half_rush_e:
            lr_e = ((1 - math.cos(x * math.pi / rush_down_epoch)) / 2) * (y1/10 - y1) + y1
        else:
            lr_e = (- math.cos((x-half_rush_e) * math.pi / (steps-half_rush_e)/2 + math.pi/2)) * (y2 - 11/20*y1) + 11/20*y1
        return lr_e
    return func

def soft_cos_log(y1=0.0, y2=1.0, steps=100):
    # 1/2 epoch * cos(0, pi) + x*(1- logx)
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    def func(x):
        rush_down_epoch = 300
        half_rush_e = round(rush_down_epoch/2)
        if x <= half_rush_e:
            lr_e = ((1 - math.cos(x * math.pi / rush_down_epoch)) / 2) * (y1/10 - y1) + y1
        else:
            lr_e = (((x-half_rush_e)/ (steps-half_rush_e))*(1 - math.log(((x-half_rush_e)/ (steps-half_rush_e)))))* (
                        y2 - 11 / 20 * y1) + 11 / 20 * y1
        return lr_e
    return func

def log_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x:1 + x/steps * (math.log(x/steps+1e-16) -1)*(y1-y2)+y2
