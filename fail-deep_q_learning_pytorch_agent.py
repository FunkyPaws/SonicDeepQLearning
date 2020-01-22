import retro
import numpy as np
import random
import math

from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

print(retro.__path__)

Experience = namedtuple('Experience', ('state', 'action', 'new_state', 'reward'))


class DQN(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()

        self.fc1 = nn.Linear(in_features=img_height * img_width * 3, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=12)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
            self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    # exploration vs exploitation
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)


class Agent:
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            return torch.tensor([random.randrange(self.num_actions)]).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)


class EnvManager:
    def __init__(self, device):
        self.device = device
        self.env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1.test')
        self.env.reset()
        self.current_screen = None
        self.done = False

    def render(self, mode="human"):
        return self.env.render(mode)

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def num_actions_availible(self):
        return self.env.action_space.n

    def take_action(self, action):
        new_state, reward, self.done, info = self.env.step(action)
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.current_screen is None

    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_widht(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        # print("~~~~~~~~~~~ screen: ", screen)
        screen_hight = screen.shape[1]

        # strip of the top and bottom part
        top = int(screen_hight * 0.1)
        bottom = int(screen_hight * 0.9)
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        resize = T.Compose([T.ToPILImage(), T.Resize((40, 90)), T.ToTensor()])
        return resize(screen).unsqueeze(0).to(self.device)


class QValues:
    device = torch.device("cuda")
    # if torch.cuda.is_available() else "cpu"
    print(device)
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_loactions = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_loactions == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values


def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.new_state)

    return t1, t2, t3, t4


# variables
number_episodes = 10_000
batch_size = 250
gamma = 0.999
learning_rate = 0.001
epsilon_start = 1
epsilon_end = 0.01
epsilon_decay = 0.001
target_update = 10
memory_size = 10_000

# test?
device = torch.device("cuda")
print("current device: ",torch.cuda.current_device())
print("device name: ", torch.cuda.get_device_name(0))

# if torch.cuda.is_available() else "cpu"
print(device)
em = EnvManager(device)
strategy = EpsilonGreedyStrategy(epsilon_start, epsilon_end, epsilon_decay)
agent = Agent(strategy, em.num_actions_availible(), device)
memory = ReplayMemory(memory_size)

policy_net = DQN(em.get_screen_height(), em.get_screen_widht()).to(device)
target_net = DQN(em.get_screen_height(), em.get_screen_widht()).to(device)
target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=learning_rate)

episodes_durations = []
for episode in range(number_episodes):
    em.reset()
    state = em.get_state()

    for timestep in count():
        em.render()

        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            # print("henk happend")
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + reward

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del loss

        if em.done:
            episodes_durations.append(timestep)
            break

        # if episode % target_update == 0:
        #     target_net.load_state_dict(policy_net.state_dict())

em.close()
