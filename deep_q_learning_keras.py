import retro
import numpy as np
import random
import cv2
import os

from keras.models import load_model, Sequential
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Flatten, Dense
from collections import deque



BUFFER_SIZE = 5_000
MINIBATCH_SIZE = 500

TOT_FRAME = 5_000
EPOCH = 10_000

MIN_OBSERVATION = 2500

DECAY_RATE = 0.99
EPSILON_DECAY = 1_000_000
FINAL_EPSILON = 0.01
INITIAL_EPSILON = 0.5

NUM_ACTIONS = 12
TAU = 0.01
# Number of frames to throw into network
NUM_FRAMES = 3
new_size_width = 90
new_size_height = 90


class ReplayBuffer:
    """Constructs a buffer object that stores the past moves
    and samples a set of subsamples"""

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, state, action, reward, done, new_state):
        """Add an experience to the buffer"""
        experience = (state, action, reward, done, new_state)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample(self, batch_size):
        """Samples a total of elements equal to batch_size from buffer
        if buffer contains enough elements. Otherwise return all elements"""

        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        # Maps each experience in batch in batches of states, actions, rewards
        # and new states
        state_batch, action_batch, reward_batch, done_batch, new_date_batch = list(map(np.array, list(zip(*batch))))

        return state_batch, action_batch, reward_batch, done_batch, new_date_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


class DeepQ(object):
    """Constructs the desired deep q learning network"""

    def __init__(self):
        self.construct_q_network()

    def construct_q_network(self):
        # Uses the network architecture found in DeepMind paper: https://arxiv.org/pdf/1312.5602v1.pdf
        self.model = Sequential()
        self.model.add(Conv2D(32, 8, 8, subsample=(4, 4), input_shape=(new_size_width, new_size_height, NUM_FRAMES)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, 4, 4, subsample=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(NUM_ACTIONS))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.00001))

        # Creates a target network as described in DeepMind paper: https://arxiv.org/pdf/1312.5602v1.pdf
        self.target_model = Sequential()
        self.target_model.add(Conv2D(32, 8, 8, subsample=(4, 4), input_shape=(new_size_width, new_size_height, NUM_FRAMES)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Conv2D(64, 4, 4, subsample=(2, 2)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Conv2D(64, 3, 3))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Flatten())
        self.target_model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.target_model.add(Dense(NUM_ACTIONS))
        self.target_model.compile(loss='mse', optimizer=Adam(lr=0.00001))

    def predict_movement(self, data, epsilon):
        q_actions = self.model.predict(data.reshape(1, new_size_width, new_size_height, NUM_FRAMES), batch_size=1)
        print(q_actions)
        opt_policy = np.argmax(q_actions)
        # print(opt_policy)
        rand_val = np.random.random()
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, NUM_ACTIONS)
        return opt_policy, q_actions

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        """Trains network to fit given parameters"""
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, NUM_ACTIONS))

        for i in range(batch_size):
            # print(s_batch[i])
            targets[i] = self.model.predict(s_batch[i].reshape(1, new_size_width, new_size_height, NUM_FRAMES), batch_size=1)
            fut_action = self.target_model.predict(s2_batch[i].reshape(1, new_size_width, new_size_height, NUM_FRAMES), batch_size=1)
            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] += DECAY_RATE * np.max(fut_action)

        loss = self.model.train_on_batch(s_batch, targets)

        # Print the loss every 10 iterations.
        if observation_num % 10 == 0:
            print("We had a loss equal to ", loss)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        self.model = load_model(path)
        print("Succesfully loaded network.")

    def target_train(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)


class Sonic(object):

    def __init__(self):
        self.env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1.test', scenario='Custom_scenario')
        self.env.reset()
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.deep_q = DeepQ()

        self.env.render()

        # A buffer that keeps the last 3 images
        self.process_buffer = []
        # Initialize buffer with the first frame
        s1, r1, _, _ = self.env.step([0])
        s2, r2, _, _ = self.env.step([0])
        s3, r3, _, _ = self.env.step([0])
        self.process_buffer = [s1, s2, s3]

    def convert_process_buffer(self):
        """Converts the list of NUM_FRAMES images in the process buffer
        into one training sample"""
        black_buffer = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (new_size_width, new_size_height)) for x in self.process_buffer]
        henk = np.array(black_buffer)
        # print("henk: ", henk.shape)
        henk1 = np.transpose(henk, (2, 1, 0))
        # print("henk1: ", henk1.shape, type(henk1))
        return henk1

    def train(self, num_frames):
        observation_num = 0
        curr_state = self.convert_process_buffer()
        epsilon = INITIAL_EPSILON
        alive_frame = 0
        total_reward = 0

        while observation_num < num_frames:
            self.env.render()
            if observation_num % 1000 == 999:
                print(("Executing loop %d" % observation_num))

            initial_state = self.convert_process_buffer()
            self.process_buffer = []

            predict_movement, predict_q_value = self.deep_q.predict_movement(curr_state, epsilon)
            print("predicted move: ",predict_movement)
            # print([predict_movement])
            # print(predict_q_value)
            movement  = [0,0,0,0,0,0,0,0,0,0,0,0]
            if predict_movement in [0, 8]:
                predict_movement = 7

            movement[predict_movement] = 1
            # print(movement)

            reward, done = 0, False
            for i in range(NUM_FRAMES):
                temp_observation, temp_reward, temp_done, info = self.env.step(movement)
                # print(info)
                reward += temp_reward
                self.process_buffer.append(temp_observation)
                done = done | temp_done

            if observation_num % 10 == 0:
                print("We predicted a q value of ", movement)

            if done:
                # print("Lived with maximum time ", alive_frame)
                print("Earned a total of reward equal to ", total_reward)
                self.env.reset()
                self.replay_buffer.clear()
                alive_frame = 0
                total_reward = 0

            new_state = self.convert_process_buffer()
            self.replay_buffer.add(initial_state, np.argmax(movement), reward, done, new_state)
            total_reward += reward

            if self.replay_buffer.size() > MIN_OBSERVATION:
                state_batch, action_batch, reward_batch, done_batch, new_date_batch = self.replay_buffer.sample(MINIBATCH_SIZE)
                self.deep_q.train(state_batch, action_batch, reward_batch, done_batch, new_date_batch, observation_num)
                self.deep_q.target_train()
                self.replay_buffer.clear()

                # Slowly decay the learning rate
                if epsilon > FINAL_EPSILON:
                    epsilon -= FINAL_EPSILON
                    # epsilon -= (FINAL_EPSILON - INITIAL_EPSILON ) / EPSILON_DECAY
                    # print("epsilon: ", epsilon)

            # Save the network every n iterations
            if observation_num % 2_500 == 2_499:
                print("Saving Network")
                self.deep_q.save_network("saved.h5")

            alive_frame += 1
            observation_num += 1



if __name__ == "__main__":
    sonic = Sonic()
    if os.path.isfile("saved.h5"):
        sonic.deep_q.load_network("saved.h5")
        print("h5 loaded")

    for i in range(EPOCH):
        sonic.train(TOT_FRAME)
        print("~~~~~~~~~~ Epoch done ~~~~~~~~~~")
