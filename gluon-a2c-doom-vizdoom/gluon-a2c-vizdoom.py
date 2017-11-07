# M.Haneferd, based on E. Culurciello: learning_pytorch.ph from August 2017
#

import itertools as it
import skimage.color, skimage.transform
from vizdoom import *
from time import sleep
import numpy as np
import mxnet as mx
import mxnet.ndarray as F
from mxnet import autograd
from mxnet import gluon

EPISODES = 500000  # Number of episodes to be played
LEARNING_STEPS = 250  # Maximum number of learning steps within each episodes
DISPLAY_COUNT = 1000  # The number of episodes to play before showing statistics.

gamma = 0.99
learning_rate = 0.0005

# Other parameters
frame_repeat = 12
resolution = (30, 45)

model_savefile = "./model-doom.pth"

ctx = mx.cpu()

# Configuration file path
# config_file_path = "../../scenarios/simpler_basic.cfg"
# config_file_path = "../../scenarios/rocket_basic.cfg"
config_file_path = "../../ViZDoom/scenarios/basic.cfg"
# config_file_path = "../../scenarios/deathmatch.cfg"

manualSeed = 1  # Set the desired seed to reproduce the results
mx.random.seed(manualSeed)

# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

# gluon.Block is the basic building block of models.
# You can define networks by composing and inheriting Block:
class Net(gluon.Block):
    def __init__(self, available_actions_count):
        super(Net, self).__init__()
        with self.name_scope():
            self.conv1 = gluon.nn.Conv2D(8, kernel_size=6, strides=3)
            self.conv2 = gluon.nn.Conv2D(8, kernel_size=3, strides=2)
            self.dense = gluon.nn.Dense(200, activation='relu')
            self.dense2 = gluon.nn.Dense(200, activation='relu')
            self.action_pred = gluon.nn.Dense(available_actions_count, in_units=200)
            self.value_pred = gluon.nn.Dense(1, in_units=200)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape((-1, 192))
        x = self.dense(x)
        x = self.dense2(x)
        probs = self.action_pred(x)
        values = self.value_pred(x)
        return mx.ndarray.softmax(probs), values


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
    doom_actions = [list(a) for a in it.product([0, 1], repeat=n)]

    loss = gluon.loss.L2Loss()
    model = Net(len(doom_actions))
    model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    optimizer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': learning_rate})

    print("Start the training!")
    episode_rewards = 0
    final_rewards = 0

    running_reward = 10 # Usikker
    train_episodes_finished = 0
    train_scores = []
    close_game = False
    for episode in range(0, EPISODES):
        game.new_episode()


        if episode > (EPISODES - 11):  # Show last 10 episodes
            sleep(1.0)  # Sleep between episodes
            if not close_game:
                close_game = True
                game.close()
                input("Press Enter to watch the last 10 episodes live... \
(If using PyCharm, make sure you have focus in this window.)")

            game.set_window_visible(True)
            game.set_mode(Mode.ASYNC_PLAYER)
            game.init()
            show_game = True

        s1 = preprocess(game.get_state().screen_buffer)
        s1 = s1.reshape([1, 1, resolution[0], resolution[1]])
        s1 = mx.nd.array(s1)
        s1 = s1.as_in_context(ctx)

        rewards = []
        values = []
        actions = []
        heads = []

        with autograd.record():
            for learning_step in range(LEARNING_STEPS):
                # Converts and down-samples the input image

                prob, value = model(s1)

                index, logp = mx.nd.sample_multinomial(prob, get_prob=True)
                action = index.asnumpy()[0].astype(np.int64)

                #  This is only to make the last games nice to watch:
                if close_game:
                    reward = game.make_action(doom_actions[action])
                    for _ in range(frame_repeat):
                        game.advance_action()
                else:
                    reward = game.make_action(doom_actions[action], frame_repeat)

                isterminal = game.is_episode_finished()

                rewards.append(reward)
                actions.append(action)
                values.append(value)
                heads.append(logp)

                if isterminal:
                    score = game.get_total_reward()
                    train_scores.append(score)
                    break
                s1 = preprocess(game.get_state().screen_buffer) if not isterminal else None
                s1 = s1.reshape([1, 1, resolution[0], resolution[1]])
                s1 = mx.nd.array(s1)
                s1 = s1.as_in_context(ctx)

            # reverse accumulate and normalize rewards
            R = 0
            for i in range(len(rewards) - 1, -1, -1):
                R = rewards[i] + gamma * R
                rewards[i] = R
            rewards = np.array(rewards)
            rewards -= rewards.mean()
            rewards /= rewards.std() + np.finfo(rewards.dtype).eps

            # compute loss and gradient
            L = sum([loss(value, mx.nd.array([r])) for r, value in zip(rewards, values)])
            final_nodes = [L]
            for logp, r, v in zip(heads, rewards, values):
                reward = r - v.asnumpy()[0, 0]
                # Here we differentiate the stochastic graph, corresponds to the
                # first term of equation (6) in https://arxiv.org/pdf/1506.05254.pdf
                # Optimizer minimizes the loss but we want to maximizing the reward,
                # so use we use -reward here.
                final_nodes.append(logp * (-reward))
            autograd.backward(final_nodes)
        optimizer.step(s1.shape[0])


        if episode % DISPLAY_COUNT == 0:
            train_scores = np.array(train_scores)
            print("Episodes {}\t".format(episode),
                  "Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
            train_scores = []

    game.close()
