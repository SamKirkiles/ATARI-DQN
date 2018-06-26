# ATARI-DQN

### Beating classic ATARI games with Deep Q Learning.

![](https://media.giphy.com/media/9VcPcvjfw9xc2o3sgt/giphy.gif)
![](https://i.imgur.com/crYtWLq.png)

Download the saves folder here: https://www.dropbox.com/sh/260ywdeeker4k3e/AABLTa2OPg0PYfV6i4vxnIVAa?dl=0
To use the saves, place the folder inside the directory and rename it to "saves". Make sure to set restore to True in main.py.

Video demo: https://www.youtube.com/watch?v=iqdxMtFzw70

**Important** Due to reward scaling to region [-1,1] the reward graph shows the number of blocks destroyed every life. Score with the current weights is around 70 but will be much more with more training.

Hyperparameters:
Due to some inconsistencies with the original Nature paper, I had to play around a bit to find good hyperparameters. To replicate the results on your own implementation, pay attention to these sections:

```
selected_actions = tf.reduce_sum(self.dense4 * tf.one_hot(self.actions,self.nA),axis=1)

self.losses = tf.losses.huber_loss(labels=self.targets,predictions=selected_actions,delta=1.0)
self.loss = tf.reduce_mean(self.losses)

self.optimizer= tf.train.RMSPropOptimizer(0.00025,momentum=0.95,epsilon=0.01)
```
I used huber loss with a delta of 1.0 (an argument was made online to use a delta of 2.0 however rewards on breakout are never -1). My momentum was 0.95. and epsisilon was 0.01. Learning rate decay was the default at 0.9. Make sure to clip rewards or they will not work with huber loss. 

To train, run with correct settings in main.py:

```
python3 main.py
```
The program will create a folder called monitor in the directory and will output videos of each episode in .mp4 file types. You can also add ```env.render()``` in the main loop to see training in real time. 

TODO:

Add testing loop
Finish Training

Links:

https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

