import gym 
import pylab
import numpy as np

import plotly.express as px
import cufflinks as cf
cf.go_offline(connected=True)

from maxent import *
from environment import Moduleviser

n_states = 75 * 75
n_actions = 400

q_table = np.zeros((n_states, n_actions))
feature_matrix = np.eye((n_states))

gamma = 0.99
q_learning_rate = 0.03
theta_learning_rate = 0.05

np.random.seed(1)

def idx_demo():
    raw_demo = np.load(file="expert_demo/expert_trajectories.npy")
    demonstrations = np.zeros((len(raw_demo), len(raw_demo[0]), 3))

    for x in range(len(raw_demo)):
        for y in range(len(raw_demo[0])):
            demonstrations[x][y][1] = raw_demo[x][y][-1]
            a_1, a_2 = 0, 0
            for z in range(len(raw_demo[0][0])-1):
                row, col = divmod(z,20)
                if ((row>=10 and row<=19)and(col>=0 and col <=4))or((row>=5 and row<=9)and(col>=15 and col<=19)):
                    if raw_demo[x][y][z] == 1:
                        a_1 += 1
                    if ((row>=5 and row <=9)and(col >=0 and col <=4))or((row>=5 and row <=9)and(col >=10 and col<=14))or((row>10 and row<=14)and(col>=15 and col<=19)):
                        if raw_demo[x][y][z] == 1:
                            a_2 += 1
            state_idx = 75*a_1 + a_2

            demonstrations[x][y][0] = state_idx
    return demonstrations

def idx_state(state):
    a_3, a_4 = 0, 0
    for x in range(len(state)):
        for y in range(len(state[0])):
            if ((x>=10 and x<=19)and(y>=0 and y<=4))or((x>=5 and x<=9)and(y>=15 and y<=19)):
                if state[x][y] == 1:
                    a_3 += 1
            if ((x>=5 and x<=9)and(y>=0 and y<=4))or((x>=5 and x<=9)and(y>=10 and y<=14))or((x>=10 and x<=14)and(y>=15 and y<=19)):
                if state[x][y] == 1:
                    a_4 += 1
    state_idx = 75*a_3 + a_4
    return state_idx

def update_q_table(state, action, reward, next_state):
    q_1 = q_table[state][action] # previous value
    q_2 = reward + gamma * max(q_table[next_state]) # q_value calculation based on bellman equation
    q_table[state][action] += q_learning_rate * (q_2-q_1)

def main():
    env = Moduleviser()
    demonstrations = idx_demo()

    expert = expert_feature_expectations(feature_matrix, demonstrations)
    learner_feature_expectations = np.zeros(n_states)
    theta = -(np.random.uniform(size=(n_states,)))
    episodes, scores = [], []

    for episode in range(30000):
        state = env.reset()
        score = 0

        if (episode != 0 and episode == 10000) or (episode > 10000 and episode % 5000 == 0):
            learner = learner_feature_expectations / episode
            maxent_irl(expert, learner, theta, theta_learning_rate)

        while True:
            state_idx = idx_state(state)
            action = np.argmax(q_table[state_idx])
            next_state, CE, reward, done, _ = env.step(action)

            irl_reward = get_reward(feature_matrix, theta, n_states, state_idx)
            next_state_idx = idx_state(next_state)
            update_q_table(state_idx, action, irl_reward, next_state_idx)
            learner_feature_expectations += feature_matrix[int(state_idx)]

            score += reward
            state = next_state

            if done:
                scores.append(score)
                episodes.append(episode)
                break
        
        if episode % 1000 == 0:
            score_avg = np.mean(scores)
            print('{} episode score is {:.2f}'.format(episode, score_avg))
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./learning_curves/maxent_30000.png")
            np.save("./results/maxent_q_table", arr=q_table)
    fig = px.imshow(state, text_auto=True, labels=dict(x="components", y="components", color="I/F"),
            x=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17','18', '19'],
            y=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17','18', '19'])
    fig.show()
    print("CE:",CE)

if __name__ == '__main__':
    main()