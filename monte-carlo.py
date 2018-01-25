# monte-carlo.py
#
# Electron drift velocity simulated using Monte Carlo methods
# For ELEC 4700, January 2018
# David Lister
#

import random
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
mass = 9.10938356e-31 #kg
potential = 10 # V
length = 1 # m
mean_free_path = 0.1 # m
charge = 1.60217662e-19 # C
n_charges = 1000


# Calculations
field_strength = potential/length #v/m
acceleration = field_strength * charge / mass
distance_to_time = lambda d: np.sqrt(2 * d / acceleration)
delta_v = lambda dt: acceleration * dt


# Functions
def get_distance(mfp):
    # Returns a random distance given a mean free path
    val = random.random()
    if val == 0:
        val = 1e-4
    val = val + 0j
    return mfp * np.real(np.log(-1/val))

def case_to_velocity_series(case, time_series):
    step = time_series[1] - time_series[0]
    tally = 0
    event_queue = [tally]
    for time in case[1]:
        tally += time
        event_queue.append(tally)
    index = 0
    velocity = []
    for t in time_series:
        if index < len(event_queue):
            if event_queue[index] <= t:
                velocity.append(0)
                index += 1
            else:
                velocity.append(velocity[-1] + delta_v(step))
        else:
            velocity.append(velocity[-1] + delta_v(step))
    return velocity



# Simulation
print("Starting")
world = []
for particle in range(n_charges):
    distance = []
    time = []
    meta = {}
    position = 0
    while position < length:
        step = get_distance(mean_free_path)
        distance.append(step)
        time.append(distance_to_time(step))
        position += step
    meta["time"] = sum(time)
    meta["distance"] = sum(distance)
    meta["avg_vel"] = meta["distance"]/ meta["time"]
    world.append([distance, time, meta])

vel = []
for case in world:
    vel.append(case[2]["avg_vel"])
drift_velocity = np.mean(vel)
drift_stdev = np.std(vel)

print("Drift velocity is: ", drift_velocity, " and the error is +- ", drift_stdev)

time = min([case[2]["time"] for case in world])
t = np.linspace(0, time, 3000)

ylst = []
for case in world:
    ylst.append(case_to_velocity_series(case, t))
ylst = np.array(ylst)
avg_vel = [np.mean(row) for row in ylst.transpose()]
plt.plot(t, avg_vel)
plt.show()


