# @Time : 2019/1/21 9:03 PM 
# @Author : Kaishun Zhang 
# @File : two-dimensional search.py 
# @Function: 二维寻宝藏
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

N_STATES = 16   # the length of the 1 dimensional world
ACTIONS = ['up', 'down', 'left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.0001    # fresh time for one move

abstacle = [6, 9] # 有障碍的状态
goal = 10  # 宝藏的位置


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    # invalid move
    if S == 0 and (A == 'up' or A == 'left'):
        return None,None;
    if S == 3 and (A == 'up' or A == 'right'):
        return None,None;
    if S == 12 and (A == 'left' or A == 'down'):
        return None,None
    if S == 15 and (A == 'right' or A == 'down'):
        return None,None
    if S // 4 == 0 and A == 'up':
        return None, None
    if S % 4 == 0 and A == 'left':
        return None,None
    if S % 4 == 3 and A == 'right':
        return None,None
    if S // 4 == 3 and A == 'down':
        return None,None

    if A == 'right':
        if S + 1 == goal:
            S_ = 'terminal'
            R = 1
        elif S + 1 in abstacle:
            S_ = S + 1
            R = -1
        else:
            S_ = S + 1
            R = 0

    elif A == 'left':
        if S - 1 == goal:
            S_ = 'terminal'
            R = 1
        elif S - 1 in abstacle:
            S_ = S - 1
            R = -1
        else:
            S_ = S - 1
            R = 0

    elif A == 'up':
        if S - 4 == goal:
            S_ = 'terminal'
            R = 1
        elif S - 4 in abstacle:
            S_ = S - 4
            R = -1
        else:
            S_ = S - 4
            R = 0

    elif A == 'down':
        if S + 4 == goal:
            S_ = 'terminal'
            R = 1
        elif S + 4 in abstacle:
            S_ = S + 4
            R = -1
        else:
            S_ = S + 4
            R = 0

    return S_, R


def update_env(S, episode, step_counter):
    print('\r', end = '')
    if S == 0:
        print('beginning')
    print('----------------------- ')
    if S == 'terminal':
        print('\nEpisode %s: total_steps = %s' % (episode + 1, step_counter))
    for i in range(N_STATES):
        if i == S :
            print('0 ',end = '')
            if i % 4 == 3:
                print('')
            continue
        if i in abstacle:
            print('-1 ',end = '')
            if i % 4 == 3:
                print('')
            continue
        if i == goal :
            print('1 ',end = '')
            if i % 4 == 3:
                print('')
            continue
        print('-2 ',end = '')
        if i % 4 == 3:
            print('')
    time.sleep(FRESH_TIME)
    if S == 'terminal':
        print('have find')
    print('\r', end = '')



def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            while(S_ is None and R is None): #
                A = choose_action(S, q_table)
                S_, R = get_env_feedback(S, A)

            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state
            if not S in abstacle:
                update_env(S, episode, step_counter+1)
            else:
                S = 0
                print('endding ------ ')
            step_counter += 1
    return q_table


if __name__== '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)