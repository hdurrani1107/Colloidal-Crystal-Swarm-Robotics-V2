#######################################################################
# schedule.py
#
# temperature schedules
#
# Author: Humzah Durrani 
#######################################################################

##########################
# Importing Libraries
##########################

import numpy as np

##########################
#Up down Temperature Schedule
##########################
def temp_schedule_1(max_temp, n_frames):
    half_frames = n_frames // 2
    temperature_schedule = np.concatenate([
        np.linspace(0, max_temp, half_frames),
        np.linspace(max_temp, 0, n_frames - half_frames - 500),
        np.full(500,0)
    ])
    return temperature_schedule
 
##########################
#Stepped Temperature Schedule
##########################
def temp_schedule_2():
    temp_steps = np.linspace(150, 1, 20)
    frames_per_step = 500
    temperature_schedule = np.concatenate([
        np.full(frames_per_step, T) for T in temp_steps
    ])
    return temperature_schedule

##########################
#Cooling Schedule 1
##########################
def temp_schedule_3(max_temp, n_frames):
    quarter_frames = int(n_frames / 4)
    #print(quarter_frames)
    temperature_schedule = np.concatenate([
        np.linspace(max_temp, 100, quarter_frames),
        np.linspace(100, 50, quarter_frames),
        np.linspace(50,0, quarter_frames),
        np.full(quarter_frames,0)
    ])
    return temperature_schedule

##########################
#Cooling Schedule 1
##########################
def temp_schedule_4():
    temperature_schedule = np.concatenate([
        np.full(5000, 15),
        np.full(5000, 10),
        np.full(5000, 5),
        np.full(5000, 0),
    ])
    return temperature_schedule