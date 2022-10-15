import math
import numpy as np
from typing import Optional

import plotly.express as px
import cufflinks as cf
cf.go_offline(connected=True)

import gym
import copy
from gym import spaces
from gym.error import DependencyNotInstalled

class Moduleviser(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None):

        self.CE = 0

        self.low = np.zeros(shape=(400,), dtype=np.int32)
        self.high = np.ones(shape=(400,), dtype=np.int32)

        self.action_space = spaces.Discrete(400)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, action):

        state = np.reshape(self.state, (20,20))

        row, col = divmod(action, 20)

        if state[row][col] == 0:
            state[row][col] = 1
            state[col][row] = 1
        else:
            state[row][col] = 0
            state[col][row] = 0
        
        self.state, self.CE = self.clustering(state)
        done = bool(self.CE >= 0.03)
        reward = -1

        return np.array(self.state, dtype=np.int32), self.CE, reward, done, {}

    def reset(self):
        state = []
        for i in range(20):
            state.append([0])
        
        state[0]  = [1,0,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0]
        state[1]  = [0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0]
        state[2]  = [1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0]
        state[3]  = [1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0]
        state[4]  = [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        state[5]  = [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        state[6]  = [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        state[7]  = [1,1,0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0]
        state[8]  = [0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0]
        state[9]  = [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0]
        state[10] = [1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0]
        state[11] = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1]
        state[12] = [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0]
        state[13] = [0,0,1,1,0,0,0,0,0,1,1,0,1,1,1,0,1,1,0,0]
        state[14] = [0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,1,0,1,0]
        state[15] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        state[16] = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0]
        state[17] = [0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1,0,0]
        state[18] = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1]
        state[19] = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1]

        state = np.array(state)
        self.state = np.reshape(state, (-1,1))

        return self.state, {}

    def clustering(self, state):
        
        # initialize weight matrix
        w = np.zeros((5,len(state[1])))
        for i in range(len(w)):
            for j in range(len(w[1])):
                w[i][j] = np.random.random()

        # initialize result matrix
        cluster_result = np.zeros((5,20))
        learning_rate = 0.03

        #calculate Euclidean distances between each row and weight vectors
        for i in range(len(state)): # fix one row at state matrix
            distances = np.zeros(len(w))
            for j in range(len(distances)):
                dist = 0
                for k in range(len(state[1])):
                    dist += (state[i][k] - w[j][k]) ** 2
                distances[j] = dist
            idx = np.argmax(distances)
            cluster_result[idx][i] = 1

            # update the winning neuron
            for m in range(len(state[1])):
                w[idx][m] += learning_rate * (state[i][m] - w[idx][m])

            # update the neighbor neurons
            if idx == 0:
                w[1][m] += learning_rate * (state[i][m] - w[idx][m])

            elif idx == 1:
                w[0][m] += learning_rate * (state[0][m] - w[0][m])
                w[2][m] += learning_rate * (state[2][m] - w[2][m])

            elif idx == 2:
                w[1][m] += learning_rate * (state[1][m] - w[1][m])
                w[3][m] += learning_rate * (state[3][m] - w[3][m])
            
            elif idx == 3:
                w[2][m] += learning_rate * (state[2][m] - w[2][m])
                w[4][m] += learning_rate * (state[4][m] - w[4][m])

            else:
                w[3][m] += learning_rate * (state[3][m] - w[3][m])

        a = 0.5
        b = 0.5

        classify_components_into_modules = cluster_result.sum(axis = 1)

        sorted_classify_components_into_modules = copy.deepcopy(classify_components_into_modules)
        sorted_classify_components_into_modules.sort()

        reversed = sorted_classify_components_into_modules[::-1]

        new_order = []
        check = np.ones(shape=(20,), dtype=np.int32)
        for i in range(len(reversed)):
            for j in range(len(classify_components_into_modules)):
                if reversed[i] == classify_components_into_modules[j]:
                    for k in range(20):
                        if cluster_result[j][k] == 1 and check[k] == 1:
                            new_order.append(k)
                            check[k] = 0
        
        clustered_matrix = np.eye(20, dtype=np.int32)

        # make new DSM matrix
        for i in range(len(clustered_matrix)):
            for j in range(i, len(clustered_matrix)):
                if state[i][j] == 1:
                    clustered_matrix[new_order.index(i)][new_order.index(j)] = 1
                    clustered_matrix[new_order.index(j)][new_order.index(i)] = 1

        # initialize the S_in and S_out
        S_in = 0
        for i in range(len(classify_components_into_modules)):
            S_in += 0.5 * classify_components_into_modules[i] * (classify_components_into_modules[i] - 1)
        S_out = 0

        for i in range(len(state)):
            for j in range(i+1,len(state[0])):
                if state[i][j] == 1:
                    # comp_i and comp_j are in same module
                    if [row[i] for row in cluster_result] == [row[j] for row in cluster_result]: 
                        S_in -= 1
                        
                    else: # comp_i and comp_j are not in same module
                        S_out += 1
                        
                        
        CE = 1 / (a * S_in + b * S_out)

        return clustered_matrix, CE 