#  The following python script is made with a combination of ascii environment and a2c model
#  created with mxnet gluon.
#  I found it difficult to Vizdoom work, so I created a "Ascii-Doom".
#
#  Example output:
#  x----+-Episodes 125000	 Results: mean: 64.5 +/- 80.3, min: -131.0, max: 100.0,
#
#  Based on this python script:
#  https://github.com/apache/incubator-mxnet/blob/master/example/gluon/actor_critic.py
#

import random
import numpy as np
from time import sleep
import mxnet as mx
import mxnet.gluon as gluon
from mxnet import nd, autograd


ACTIONS = ['left', 'right', 'shoot']  # available actions
ENV_SIZE = 5  # This is the size of the environment (Number of characters to move around)
EPISODES = 300000  # Number of episodes to be played
LEARNING_STEPS = 250  # Maximum number of learning steps within each episodes
MAX_SHOTS = 5  # Max shots the player can take.
DISPLAY_COUNT = 1000  # The number of episodes to play before showing statistics and last played game.

#  If you have NVIDIA or are using AWS, you lucky guy, please try out gpu settings:
ctx = mx.cpu()
gamma = 0.99

#  Class for handling environment.
#  It creates a line of "-" at the length (ENV_SIZE). It places a target "x" in a random place.
#  The player is located as an "o" in the middel of the environment, which moves around.
#  If the player shoots, it is marked with a "+"
#
#  The environment consists of two rows in an array of size ( 2, ENV_SIZE). The player moves
#  in the lowest row, and the target are placed in the upmost row.
#  Example:
#  ------x--    : Target
#     o         : Player
#
#  Since the terminal only can show movement in one line without "\n", the display combines the array in the output
#  Example:
#  ---o--x--    : Target and player combined to get a good output
#

class env:
    def __init__(self, size, max_steps, max_shots):
        self.size = size
        self.env_list = []
        self.env_player = []
        self.window = []
        self.max_steps = max_steps
        self.max_shots = max_shots

        # Initialize environment:
        self.target = random.randint(0, self.size - 1)
        self.position = int(self.size / 2)
        self.total_reward = 0
        self.num_steps = 0
        self.num_shots = 0
        self.env_list = ['-'] * (self.size)  # '----x-----' our environment
        self.env_list[self.target] = 'x'
        self.window = ['-'] * (self.size)
        self.window[self.target] = 'x'
        self.window[self.position] = 'o'
        self.env_player = [' '] * (self.size)  # '....o.....' Player environment
        self.env_player[self.position] = 'o'

    def new_game(self):
        # Initialize environment:
        self.target = random.randint(0, self.size - 1)
        self.position = int(self.size / 2)
        self.total_reward = 0
        self.num_steps = 0
        self.num_shots = 0
        self.env_list = ['-'] * (self.size)  # '----x-----' our environment
        self.env_list[self.target] = 'x'
        self.window = ['-'] * (self.size)
        self.window[self.target] = 'x'
        self.window[self.position] = 'o'
        self.env_player = [' '] * (self.size)  # '....o.....' Player environment
        self.env_player[self.position] = 'o'

    def make_action(self, A):
        # This is how agent will interact with the environment
        terminal = False

        self.num_steps = self.num_steps + 1 # Count number of steps

        if A == 'right':  # move right
            self.position = self.position + 1
            R = -1  # Move Score
            if self.position == self.size:
                self.position = self.position - 1  # At the end.

        if A == 'left':  # move left
            R = -1
            if self.position > 0:
                self.position = self.position - 1

        if A == 'shoot':  # shoot
            if self.position == self.target:
                R = 100
                terminal = True  # terminate due to killed the target
            else:
                R = -25
                self.num_shots = self.num_shots + 1
                if self.max_shots == self.num_shots:
                    terminal = True  # terminate due to empty magazine

        self.total_reward = self.total_reward + R

        # End game if maximum number of actions performed has happened
        if self.num_steps == self.max_steps:
            terminal = True

        return terminal, self.get_env(), R

    def update_env(self, terminal, A, episode, step_counter, display = False):
        # This is how environment be updated
        self.env_list = ['-'] * (self.size)  # '----x-----' our environment
        self.env_list[self.target] = 'x'
        self.window = ['-'] * (self.size)
        self.window[self.target] = 'x'
        self.window[self.position] = 'o'
        self.env_player = [' '] * (self.size)  # '....o.....' Player environment
        self.env_player[self.position] = 'o'
        if terminal == 'terminal':
            if display:
                interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
                print('\r{}'.format(interaction), end='')
                sleep(2)
                print('\r                                ', end='')
        else:
            if A == 'shoot':
                self.window[self.position] = '+'
                interaction = ''.join(self.window)
                if display:
                    print('\r{}'.format(interaction), end='')
                    sleep(0.1)
            else:
                self.window[self.position] = 'o'
                interaction = ''.join(self.window)
                if display:
                    print('\r{}'.format(interaction), end='')
                    sleep(0.1)
        return self.window

    def get_env(self):
        ret = np.array(list(str(ord(c)) for c in self.env_list), dtype=int)
        ret = np.vstack((ret, np.array(list(str(ord(c)) for c in self.env_player), dtype=int)))
        return ret

    def get_total_reward(self):
        return self.total_reward

#  The model (Core)
#  The model receives an input with array of size ( 2, ENV_SIZE ).
#  It will move it trough two layers with tensors (And I have different activations) just for trying it.
#  Finally it will make an output for action and a value.
#  The action is returned with a softmax.

class Net(gluon.Block):
    def __init__(self, actions_count, num_hidden=200, num_layers=2, dropout=0):
        super(Net, self).__init__()
        with self.name_scope():
            self.dense = gluon.nn.Dense(200, activation='tanh')
            self.dense2 = gluon.nn.Dense(200, activation='relu')
            self.lstm = gluon.rnn.LSTM(num_hidden, num_layers, dropout=dropout, input_size=1)
            self.action_pred = gluon.nn.Dense(actions_count, in_units=40000)
            self.value_pred = gluon.nn.Dense(1, in_units=40000)

    def forward(self, x, hidden):
        x = self.dense(x)
        x = self.dense2(x)
        x, hidden = self.lstm(x, hidden)
        probs = self.action_pred(x)
        values = self.value_pred(x)
        return mx.ndarray.softmax(probs), values, hidden

    def begin_state(self, *args, **kwargs):
        return self.lstm.begin_state(*args, **kwargs)


def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


if __name__ == "__main__":

    #  Model initialization and loss method
    loss = gluon.loss.L2Loss()
    model = Net(len(ACTIONS))
    model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    optimizer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.001})

    print("\r\nStart training!\n")

    env = env(ENV_SIZE, LEARNING_STEPS, MAX_SHOTS) # Create and initialize environment
    train_scores = []

    for episode in range(0, EPISODES):

        #  Placeholders
        rewards = []
        values = []
        actions = []
        heads = []

        # Create new environment for the episode
        env.new_game()

        # Initialize LSTM
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = 200, ctx=ctx)

        s1 = env.get_env()
        s1 = s1.reshape([1, 1, 2, ENV_SIZE])
        s1 = nd.array(s1)
        s1 = s1.as_in_context(ctx)

        hidden = detach(hidden)

        with autograd.record():
            for learning_step in range(LEARNING_STEPS):

                #  Returns the value znd probabillity for action from the model
                prob, value, hidden = model(s1, hidden)

                index, logp = mx.nd.sample_multinomial(prob, get_prob=True)
                action = index.asnumpy()[0].astype(np.int64)

                isterminal, s1, reward = env.make_action(ACTIONS[action])

                if episode % DISPLAY_COUNT == 0:
                    env.update_env(isterminal, ACTIONS[action], episode, learning_step, True)
                else:
                    env.update_env(isterminal, ACTIONS[action], episode, learning_step)

                rewards.append(reward)
                actions.append(action)
                values.append(value)
                heads.append(logp)

                if isterminal:
                    score = env.get_total_reward()
                    train_scores.append(score)
                    break
                s1 = env.get_env()
                s1 = s1.reshape([1, 1, 2, ENV_SIZE])
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
