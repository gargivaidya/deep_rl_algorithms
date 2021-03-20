import gym
import numpy as np


env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 10
DISCRETE_OBS_SIZE = [20] * len(env.observation_space.high) 
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OBS_SIZE

q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OBS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low)/discrete_os_win_size
	return tuple(discrete_state.astype(np.int))
for episode in range(EPISODES) :
	if episode % SHOW_EVERY == 0:
		render = True
	else :
		render = False
	discrete_state = get_discrete_state(env.reset())

	done = False
	while not done:
		action = np.argmax(q_table[discrete_state]) #out of 3 actions
		state_, reward, done, info = env.step(action)
		new_discrete_state = get_discrete_state(state_)
		if render:
			env.render()
		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action, )]
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			q_table[discrete_state + (action, )] = new_q
		elif state_[0] >= env.goal_position :
			q_table[discrete_state + (action, )] = 0

		discrete_state = new_discrete_state
