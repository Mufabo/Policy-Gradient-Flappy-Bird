import multiprocessing as mp
import numpy as np
import time
import flappy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


class Net(nn.Module):
    # Define network parameters
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(72*100, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))  # Finally returns probability of flapping
        return x


def preprocess(state):
    # This function performs preprocessing on the state frame data
    #   Input: 288x512 matrix of integers
    #   Output: 72x100 matrix of floats

    state = state[:, :400]  # Crop lower part of screen below pipes
    state = state[::4, ::4]  # Downsample frame to 1/4 the size for efficiency
    state = state.ravel().astype(float)  # Convert to float (from int) and unravel from 2d->1d
    state -= state.mean()  # Normalise by subtracting mean and dividing by std
    state /= state.std() + np.finfo(float).eps  # Note that here I add eps in case std is 0
    return torch.from_numpy(state).float()  # Return state in format required by PyTorch


def main(num_episodes=10):

    agent = Net()
    state = torch.load('model/trained-model')
    agent.load_state_dict(state['state_dict'])
    agent.eval()

    input_queue = mp.Queue()
    output_queue = mp.Queue()

    for episode in range(num_episodes):
        p = mp.Process(target=flappy.main, args=(input_queue, output_queue))
        p.start()

        input_queue.put(True)  # This starts next episode
        output_queue.get()  # Gets blank response to confirm episode started
        input_queue.put(False)  # Input initial action as no flap
        state, reward, done = output_queue.get()  # Get initial state

        episode_reward = 0
        while not done:
            state = preprocess(state)  # Preprocess the raw state data for usage with agent
            flap_probability = agent(state)  # Forward pass state through network to get flap probability

            prob_dist = Bernoulli(flap_probability)  # Generate Bernoulli distribution with given probability
            action = prob_dist.sample()  # Sample action from probability distribution

            if action == 1:
                input_queue.put(True) # If action is 1, input True
            else:
                input_queue.put(False)  # Otherwise, False

            state, reward, done = output_queue.get()  # Get resulting state and reward from above action

            episode_reward += reward  # Increase current episode's reward counter

        print('Episode {}  || Reward: {}'.format(episode, episode_reward))

        time.sleep(2)
        p.terminate()  # Close current instance, to avoid freezing if another window is selected


if __name__ == "__main__":
    main()
