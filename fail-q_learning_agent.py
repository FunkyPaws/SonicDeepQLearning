import retro
import numpy as np
import random

print(retro.__path__)

# retro.make( Gamename, state )
# Gamename can be found in : retro/data/stable/game of choice
# State can be found in the previous folder
env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1.test')

number_episodes = 10_000
max_steps_per_episode = 1_000_000

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# to keep track of the scores over time this array will hold on to them
rewards_all_episodes = []

# first we need to know what the observation space and the action space are
print("observation space: ", type(env.observation_space))  # this shows that de box is shaped like Box(224, 320, 3)
state_space_size = env.observation_space.__sizeof__()
print("observation size discreet: ", state_space_size)
print("action space: ", env.action_space.n)  # this shows that Sonic is MultiBinary(12)
action_space_size = env.action_space.n

q_table = np.zeros((state_space_size, action_space_size))
# print(q_table)

for episode in range(number_episodes):
    state = env.reset()

    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        env.render()

        # exploration-exploitation
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        # # update Q_table
        # q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
        #                          learning_rate * (reward + discount_rate + np.max(q_table[new_state, :]))
        #
        # state = new_state
        # rewards_current_episode += reward

        if done:
            break
    exploration_rate = min_exploration_rate + \
                       (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    rewards_all_episodes.append(rewards_current_episode)

# calculation avg reward per 1k episodes
rewards_per_k_episodes = np.split(np.array(rewards_all_episodes), number_episodes / 1_000)
count = 1_000
print("~~~~~~~~~~ AVG reward per 1k episodes ~~~~~~~~~~\n")
for r in rewards_per_k_episodes:
    print(count, " : ", str(sum(r / 1_000)))
    count += 1_000

# updated q_table
print("~~~~~~~~~~ Updated Q_table ~~~~~~~~~~\n")
print(q_table)
