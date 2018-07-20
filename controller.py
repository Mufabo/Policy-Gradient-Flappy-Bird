import time
import multiprocessing as mp
import numpy as np
import flappy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt


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


def plot_durations(episode_durations, batch_size):
    # Plot curve of median survival duration for each batch
    # Adapted from http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode_durations)  # Plot individual episode durations

    # Here we calculate the median of each batch, and plot them
    num_batches = int(np.ceil(len(episode_durations)/batch_size))
    batch_medians = []
    for batch in range(num_batches):
        batch_median = np.median(episode_durations[batch*batch_size:(batch+1)*batch_size])
        batch_medians += [batch_median]*batch_size

    plt.plot(batch_medians)
    plt.yscale('log')
    plt.pause(0.001)  # Allow plot to update


def discount_rewards(r, gamma):
    # This function performs discounting of rewards by going back
    # and punishing or rewarding based upon final outcome of episode
    disc_r = np.zeros_like(r, dtype=float)
    running_sum = 0
    for t in reversed(range(0, len(r))):
        if r[t] == -1:  # If the reward is -1...
            running_sum = 0  # ...then reset sum, since it's a game boundary
        running_sum = running_sum * gamma + r[t]
        disc_r[t] = running_sum

    # Here we normalise with respect to mean and standard deviation:
    discounted_rewards = (disc_r - disc_r.mean()) / (disc_r.std() + np.finfo(float).eps)
    # Note that we add eps in the rare case that the std is 0
    return discounted_rewards


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


def main(num_batches=10000):
    # Notes:
    #   'state' is a 2d array containing a 288x512 matrix of integers (the frames are rotated by PyGame)
    #   'reward' is defined as:
    #                       +1 for scoring a point (getting through a pipe)
    #                       -1 for dying
    #   'done' is True upon dying, signifying the end of the episode

    # Define parameters:
    batch_size = 25  # Number of episodes to run before updating net parameters
    learning_rate = 1e-4  # Learning rate used to scale gradient updates
    gamma = 0.99  # Discount factor when accumulating future rewards
    mode = 'train'

    # Network initialisation:
    agent = Net()  # Define agent as an object of class Net (defined above)
    opt = optim.Adam(agent.parameters(), lr=learning_rate)
    # ^ Define optimiser as Adam with above defined learn rate.

    # Define queues for sending/receiving data from game:
    input_queue = mp.Queue()
    output_queue = mp.Queue()

    # Start game on a separate process:
    p = mp.Process(target=flappy.main, args=(input_queue, output_queue, mode))
    p.start()

    # Initialise storage variables and counters:
    episode_durations = []  # This variable is used for plotting only
    batch_log_prob, batch_rewards = [], []  # These variables are used for calculating loss
    batch_final_rewards = []  # This variable is used for deciding when to save
    best_batch_median = 0  # Variable to determine best model so far, for saving

    st = time.time()
    for batch in range(num_batches):
        for episode in range(batch_size):  # Start episode at 1 for easier batch management with % later
            input_queue.put(True)  # This starts next episode
            output_queue.get()  # Gets blank response to confirm episode started
            input_queue.put(False)  # Input initial action as no flap
            state, reward, done = output_queue.get()  # Get initial state

            episode_steps = 0  # Number of steps taken in current episode
            episode_reward = 0  # Amount of reward obtained from current episode

            while not done:
                state = preprocess(state)  # Preprocess the raw state data for usage with agent
                flap_probability = agent(state)  # Forward pass state through network to get flap probability

                prob_dist = Bernoulli(flap_probability)  # Generate Bernoulli distribution with given probability
                action = prob_dist.sample()  # Sample action from probability distribution
                log_prob = prob_dist.log_prob(action)  # Store log probability of action

                if action == 1:
                    input_queue.put(True) # If action is 1, input True
                else:
                    input_queue.put(False)  # Otherwise, False

                state, reward, done = output_queue.get()  # Get resulting state and reward from above action

                batch_log_prob.append(log_prob)  # Store the log probability for loss calculation
                batch_rewards.append(reward)  # Store the reward obtained as a result of action

                episode_reward += reward  # Increase current episode's reward counter
                episode_steps += 1  # Increase number of steps taken on current episode

            batch_final_rewards.append(episode_reward)
            episode_durations.append(episode_steps)  # Include current episode's step count for plot
            print('Batch {}, Episode {}  || Reward: {:.1f} || Steps: {} '.format(batch, episode, episode_reward, episode_steps))
            input_queue.put(True)  # Reset the game.

        #Once batch of rollouts is complete:

        plot_durations(episode_durations, batch_size)  # Update plot to include current batch
        discounted_rewards = discount_rewards(batch_rewards, gamma)  # Discount rewards with discount factor gamma

        opt.zero_grad()  # Zero gradients to clear existing data

        for i in range(len(batch_log_prob)):
            loss = -batch_log_prob[i]*discounted_rewards[i]  # Calculate negative log likelihood loss, scaled by reward
            loss.backward()  # Backpropagate to calculate gradients

        print('Updating network weights.')
        opt.step()  # Update network weights using above accumulated gradients

        batch_median = np.median(batch_final_rewards)
        # If current model has best median performance, save:
        if batch_median > best_batch_median:
            print('New best batch median {} (previously {}), saving network weights.'.format(batch_median, best_batch_median))
            best_batch_median = batch_median

            state = {
                'state_dict': agent.state_dict(),
                'optimizer': opt.state_dict(),
            }
            torch.save(state, 'model/trained-model.pt')

            # Load using:
            # state = torch.load(filepath)
            # agent.load_state_dict(state['state_dict']), opt.load_state_dict(state['optimizer'])

        else:
            print('Batch Median Reward: {}'.format(batch_median))

        print('Batch Size: {},  Time Taken: {:.2f}s'.format(batch_size,time.time()-st))
        st = time.time()
        batch_log_prob, batch_rewards = [], []  # Reset batch log probabilities and rewards
        batch_final_rewards = []  # Reset end reward for each episode in batch

    p.terminate()  # Once all episodes are finished, terminate the process.


if __name__ == "__main__":
    main()
