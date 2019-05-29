# This module is derived (with modifications) from # https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-rl-pong/trainer/helper.py
# Special thanks to:
# Yu-Han Liu https://nuget.pkg.github.com/dizcology
# Martin Görner https://github.com/martin-gorner

# Copyright 2019 Leigh Johnson

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# Python
# Lib
from gym.wrappers import Monitor
import numpy as np


def monitor_env(env, video_dir='video/'):
    env = Monitor(env, video_dir, force=True)
    return env

# helpers taken from:
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5


def preprocess_frame(frame,
                     # default OpenAI Atart Env frame size
                     downsample_factor=2,
                     # "erase" pixels with this color - use this setting to cancel background colors and reduce noise in input data
                     #ignore_rgb_values=[144, 104]
                     ):
    ''' Converts 210x160x3 uint8 frame into 8400 (105 x 80) 1D float vector '''
    # crop to a square
    # output a 1D float vector, downsampled by factor of 2 (105 x 80)
    downsampled = frame[::downsample_factor, ::downsample_factor, 0]
    # erase background pixels
    #downsampled[downsampled in ignore_rgb_values] = 0
    # everything else (enemies, blocks) just set to 1
    # downsampled[downsampled != 0] = 1

    return downsampled.astype(np.float).ravel()


def discount_rewards(r, decay_factor=0.99):
    '''
        Assign moves a reward between (-1, 1)

        Compute reward 1 => r > 0 for moves that contribute to game points
        Compute reward -1 <= r < 0 for moves that contribute to being hit by enemies

    '''
    r = np.array(r)
    # initialize discounted reward array with zeros
    discounted_r = np.zeros_like(r)

    # r[t] == sum of all rewards occurring after t
    sum_r = 0
    # initialize var that will hold sum through reverse iteration
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            # reset the sum, since this was a game boundary (pong specific!)
            sum_r = 0
        sum_r = sum_r * decay_factor + r[t]
        discounted_r[t] = sum_r
    return discounted_r.tolist()
