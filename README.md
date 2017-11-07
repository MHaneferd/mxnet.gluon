# mxnet.gluon
Some Python scripts that I have modified and extended to learn the Mxnet gluon package for deep learning, and Python coding itself.. Since therte is not much mxnet.gluon files out there, I have found some Pytorch files that I have translated to gluon.


[gluon-a2c.py](https://github.com/MHaneferd/mxnet.gluon/blob/master/gluon-a2c-ascii-env/gluon-a2c.py)<br />
This file have a simple synchronized actor critic model (A2C) which train on a ascii environment with a target and a player that shoots. You can easily modify number of shots for each episode. And also the number of episodes. The environment size is also customizable.

[gluon-a2c-vizdoom.py](https://github.com/MHaneferd/mxnet.gluon/blob/master/gluon-a2c-doom-vizdoom/gluon-a2c-vizdoom.py)<br />
This file also have a simple  synchronized actor critic model (A2C) which train on a ViZDoom environment. The Doom environment need to be installed before running. The logic of the python file is the same as with [gluon-a2c.py](https://github.com/MHaneferd/mxnet.gluon/blob/master/gluon-a2c-ascii-env/gluon-a2c.py), but it uses Convolutional network on the Doom environment. It is quite easy to experiment and debug models with the mxnet.gluon block, so feel free to send me some input if you see that the model could be made better.
The configuration of which Doom WAD file to use must be modified before running (As I cannot imagine you have the same path as I)

[gluon-a2c-vizdoom.py](https://github.com/MHaneferd/mxnet.gluon/blob/master/gluon-dqn-doom-vizdoom/gluon-dqn-vizdoom.py)<br />
This file uses DQN network model with mxnet gluon. It is translated from Pytorch file. ( I have not invented the logic here, just made it work with mxnet.gluon ). It uses Doom (Vizdoom) environment. The configuration of which Doom WAD file to use must be modified before running (As I cannot imagine you have the same path as I)
