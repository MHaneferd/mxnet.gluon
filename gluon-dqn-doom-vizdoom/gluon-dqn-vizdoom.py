# M.Haneferd, based on E. Culurciello: learning_pytorch.ph from August 2017
# Translated from pytorch

from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
from tqdm import trange
import mxnet as mx
import mxnet.ndarray as F
from mxnet import autograd
from mxnet import gluon

# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
epochs = 30
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 256

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 5

model_savefile = "./model-doom.pth"
save_model = True
load_model = False
skip_learning = False

ctx = mx.cpu()

# Configuration file path
config_file_path = "../../ViZDoom/scenarios/simpler_basic.cfg"
# config_file_path = "../../ViZDoom/scenarios/rocket_basic.cfg"
# config_file_path = "../../ViZDoom/scenarios/basic.cfg"
# config_file_path = "../../ViZDoom/scenarios/deathmatch.cfg"

manualSeed = 1  # Set the desired seed to reproduce the results
mx.random.seed(manualSeed)

# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img


class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, channels, resolution[0], resolution[1])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, 0, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, 0, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]

# Define Network
class Net(gluon.Block):
    def __init__(self, available_actions_count):
        super(Net, self).__init__()
        self.conv1 = gluon.nn.Conv2D(8, kernel_size=6, strides=3)
        self.conv2 = gluon.nn.Conv2D(8, kernel_size=3, strides=2)
        self.fc1 = gluon.nn.Dense(128, in_units=192)
        self.fc2 = gluon.nn.Dense(available_actions_count)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape((-1, 192))
        x = F.relu(self.fc1(x))
        return self.fc2(x)


criterion = gluon.loss.L2Loss()

def learn(s1, target_q):
    s1 = mx.nd.array(s1)
    target_q = mx.nd.array(target_q)
    s1 = s1.as_in_context(ctx)
    target_q = target_q.as_in_context(ctx)
    with autograd.record():
        output = model(s1)
        loss = criterion(output, target_q)
        loss.backward()
    optimizer.step(s1.shape[0])
    return loss

def get_q_values(state):
    state = mx.nd.array(state)
    state = state.as_in_context(ctx)
    return model(state)


def get_best_action(state):
    q = get_q_values(state)
    index =  mx.nd.argmax(q, axis=1)
    action = index.asnumpy()[0]
    return action.astype(np.int64)


def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)
        q = get_q_values(s2).asnumpy()
        q2 = np.max(q, axis=1)
        target_q = get_q_values(s1).asnumpy()
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
        learn(s1, target_q)


def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().screen_buffer)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        s1 = s1.reshape([1, 1, resolution[0], resolution[1]])
        a = get_best_action(s1)
    reward = game.make_action(actions[a], frame_repeat)

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    learn_from_memory()


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game


if __name__ == '__main__':
    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    if load_model:
        print("Loading model from: ", model_savefile)
        model = torch.load(model_savefile)
    else:
        # Initialize Parameters
        # A network must be created and initialized before it can be used:
        model = Net(len(actions))
        # Initialize on CPU. Replace with `mx.gpu(0)`, or `[mx.gpu(0), mx.gpu(1)]`,
        # etc to use one or more GPUs.
        model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

    optimizer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': learning_rate})

    print("Starting the training!")
    time_start = time()
    if not skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            game.new_episode()
            for learning_step in trange(learning_steps_per_epoch, leave=False):
                # Here the actual learning occur:
                perform_learning_step(epoch)
                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            print("\nTesting...")
            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    state = state.reshape([1, 1, resolution[0], resolution[1]])
                    best_action_index = get_best_action(state)

                    game.make_action(actions[best_action_index], frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f +/- %.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            print("Saving the network weigths to:", model_savefile)
            # torch.save(model, model_savefile)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()
    episode_score = 0

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            state = state.reshape([1, 1, resolution[0], resolution[1]])
            best_action_index = get_best_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        episode_score += score
        print("Total score: ", score)

    print("\nTotal Sum score: ", episode_score)
    print("Average score: ", episode_score/episodes_to_watch)