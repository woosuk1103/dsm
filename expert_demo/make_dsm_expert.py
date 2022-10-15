import numpy as np
import gym
from environment import Moduleviser

env = Moduleviser()

trajectories = []
episode_step = 0

for episode in range(10): # n_trajectories : 10
    trajectory = []
    step = 0

    env.reset()
    print("episode_step", episode_step)

    while True: 
        # env.render()
        print("step", step)

        input_row = input("Insert the row number you want to modify: ")
        input_col = input("Insert the column number you want to modify: ")

        row = int(input_row)
        col = int(input_col)

        action = row*20 + col
        state, CE, reward, done, _ = env.step(action)

        if (CE >= 0.05) or step > 19: # trajectory_length : 20
            break

        state_in_array = np.zeros(400)
        for i in range(len(state)):
            for j in range(len(state[0])):
                idx = i*20+j
                state_in_array[idx] = state[i][j]
                
        state_in_array = np.append(state_in_array, np.array(action))
        trajectory.append(state_in_array)
        step += 1

    trajectory_numpy = np.array(trajectory, float)
    print("trajectory_numpy.shape", trajectory_numpy.shape)
    episode_step += 1
    trajectories.append(trajectory)

np_trajectories = np.array(trajectories, float)
print("np_trajectories.shape", np_trajectories.shape)

np.save("expert_trajectories", arr=np_trajectories)