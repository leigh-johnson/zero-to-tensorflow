# This module is derived (with modifications) from # https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-rl-pong/trainer/task.py
# Special thanks to:
# Yu-Han Liu https://nuget.pkg.github.com/dizcology
# Martin Görner https://github.com/martin-gorner

# Copyright 2019 Leigh Johnson

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# Python
import logging
import argparse
import os
from collections import deque
# Lib
import tensorflow as tf
import numpy as np
import gym

from trainer.helpers import discount_rewards, prepro
from agents.tools.wrappers import AutoReset, FrameHistory


# Legal moves in space invaders are FIRE, RIGHT, LEFT, and DO NOTHING (NOOP or "No operation")
ACTIONS = {
    0: "NOOP",
    1: "FIRE",
    #   2: "UP",
    3: "RIGHT",
    4: "LEFT",
    #     5: "DOWN",
    #     6: "UPRIGHT",
    #     7: "UPLEFT",
    #     8: "DOWNRIGHT",
    #     9: "DOWNLEFT",
    #     10: "UPFIRE",
    #     11: "RIGHTFIRE",
    #     12: "LEFTFIRE",
    #     13: "DOWNFIRE",
    #     14: "UPRIGHTFIRE",
    #     15: "UPLEFTFIRE",
    #     16: "DOWNRIGHTFIRE",
    #     17: "DOWNLEFTFIRE",
}

MAX_MEMORY_LEN = 100000
MAX_EPOCH_MEMORY_SIZE = 10000

# We'll be pre-processing inputs into a 105 x 80 image diff (downsampled by a factor of 2) of currentframe - previousframe
OBSERVATION_DIM = 105 * 80


# MEMORY stores tuples:
# (observation, label, reward)
MEMORY = deque([], maxlen=MAX_MEMORY_LEN)


def gen():
    for m in list(MEMORY):
        yield m


def build_graph(observations):
    """Calculates logits from the input observations tensor.
    This function will be called twice: rollout and train.
    The weights will be shared.
    """
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        hidden = tf.layers.dense(
            observations, args.hidden_dim, use_bias=False, activation=tf.nn.relu)
        logits = tf.layers.dense(hidden, len(ACTIONS), use_bias=False)

    return logits


def main(args):
    args_dict = vars(args)
    logging.info('args: {}'.format(args_dict))

    with tf.Graph().as_default() as g:
        # rollout subgraph
        with tf.name_scope('rollout'):
            observations = tf.placeholder(
                shape=(None, OBSERVATION_DIM), dtype=tf.float32)

            logits = build_graph(observations)

            logits_for_sampling = tf.reshape(
                logits, shape=(1, len(ACTIONS)))

            # Sample the action to be played during rollout.
            sample_action = tf.squeeze(tf.multinomial(
                logits=logits_for_sampling, num_samples=1))

        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=args.learning_rate,
            decay=args.rmsprop_decay
        )

        # dataset subgraph for experience replay
        with tf.name_scope('dataset'):
            # the dataset reads from MEMORY
            ds = tf.data.Dataset.from_generator(
                gen, output_types=(tf.float32, tf.int32, tf.float32))
            ds = ds.shuffle(MAX_MEMORY_LEN).repeat().batch(args.batch_size)
            iterator = ds.make_one_shot_iterator()

        # training subgraph
        with tf.name_scope('train'):
            # the train_op includes getting a batch of data from the dataset, so we do not need to use a feed_dict when running the train_op.
            next_batch = iterator.get_next()
            train_observations, labels, processed_rewards = next_batch

            # This reuses the same weights in the rollout phase.
            train_observations.set_shape((args.batch_size, OBSERVATION_DIM))
            train_logits = build_graph(train_observations)

            cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=train_logits,
                labels=labels
            )

            # Extra loss when the paddle is moved, to encourage more natural moves.
            probs = tf.nn.softmax(logits=train_logits)
            move_cost = args.move_penalty * \
                tf.reduce_sum(probs * [0, 1.0, 1.0], axis=1)

            loss = tf.reduce_sum(processed_rewards *
                                 cross_entropies + move_cost)

            global_step = tf.train.get_or_create_global_step()

            train_op = optimizer.minimize(loss, global_step=global_step)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=args.max_to_keep)

        with tf.name_scope('summaries'):
            rollout_reward = tf.placeholder(
                shape=(),
                dtype=tf.float32
            )

            # the weights to the hidden layer can be visualized
            hidden_weights = tf.trainable_variables()[0]
            for h in range(args.hidden_dim):
                slice_ = tf.slice(hidden_weights, [0, h], [-1, 1])
                image = tf.reshape(slice_, [1, 80, 80, 1])
                tf.summary.image('hidden_{:04d}'.format(h), image)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
                tf.summary.scalar('{}_max'.format(
                    var.op.name), tf.reduce_max(var))
                tf.summary.scalar('{}_min'.format(
                    var.op.name), tf.reduce_min(var))

            tf.summary.scalar('rollout_reward', rollout_reward)
            tf.summary.scalar('loss', loss)

            merged = tf.summary.merge_all()

        logging.info('Number of trainable variables: {}'.format(
            len(tf.trainable_variables())))

    inner_env = gym.make('Pong-v0')
    # tf.agents helper to more easily track consecutive pairs of frames
    env = FrameHistory(inner_env, past_indices=[0, 1], flatten=False)
    # tf.agents helper to automatically reset the environment
    env = AutoReset(env)

    with tf.Session(graph=g) as sess:
        if args.restore:
            restore_path = tf.train.latest_checkpoint(args.output_dir)
            logging.info('Restoring from {}'.format(restore_path))
            saver.restore(sess, restore_path)
        else:
            sess.run(init)

        summary_path = os.path.join(args.output_dir, 'summary')
        summary_writer = tf.summary.FileWriter(summary_path, sess.graph)

        # lowest possible score after an episode as the
        # starting value of the running reward
        _rollout_reward = -21.0

        for i in range(args.n_epoch):
            logging.info('>>>>>>> epoch {}'.format(i+1))

            logging.info('>>> Rollout phase')
            epoch_memory = []
            episode_memory = []

            # The loop for actions/steps
            _observation = np.zeros(OBSERVATION_DIM)
            while True:
                # sample one action with the given probability distribution
                _label = sess.run(sample_action, feed_dict={
                    observations: [_observation]})

                _action = ACTIONS[_label]

                _pair_state, _reward, _done, _ = env.step(_action)

                if args.render:
                    env.render()

                # record experience
                episode_memory.append((_observation, _label, _reward))

                # Get processed frame delta for the next step
                pair_state = _pair_state

                current_state, previous_state = pair_state
                current_x = prepro(current_state)
                previous_x = prepro(previous_state)

                _observation = current_x - previous_x

                if _done:
                    obs, lbl, rwd = zip(*episode_memory)

                    # processed rewards
                    prwd = discount_rewards(rwd, args.reward_decay)
                    prwd -= np.mean(prwd)
                    prwd /= np.std(prwd)

                    # store the processed experience to memory
                    epoch_memory.extend(zip(obs, lbl, prwd))

                    # calculate the running rollout reward
                    _rollout_reward = 0.9 * _rollout_reward + 0.1 * sum(rwd)

                    episode_memory = []

                    if args.render:
                        _ = input('episode done, press Enter to replay')
                        epoch_memory = []
                        continue

                if len(epoch_memory) >= ROLLOUT_SIZE:
                    break

            # add to the global memory
            MEMORY.extend(epoch_memory)

            logging.info('>>> Train phase')
            logging.info('rollout reward: {}'.format(_rollout_reward))

            # Here we train only once.
            _, _global_step = sess.run([train_op, global_step])

            if _global_step % args.save_checkpoint_steps == 0:

                logging.info('Writing summary')

                feed_dict = {rollout_reward: _rollout_reward}
                summary = sess.run(merged, feed_dict=feed_dict)

                summary_writer.add_summary(summary, _global_step)

                save_path = os.path.join(args.output_dir, 'model.ckpt')
                save_path = saver.save(
                    sess, save_path, global_step=_global_step)
                logging.info('Model checkpoint saved: {}'.format(save_path))


def parse_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument(
        '--loglevel',
        type='string',
        default='INFO',
        choices=['debug', 'info', 'error', 'warning',
                 'DEBUG', 'INFO', 'ERROR', 'WARNING']
    )
    parser.add_argument(
        '--n-epoch',
        type=int,
        default=5000,
        help='Number of iterations (training rounds) to run'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Number of batches to divide dataset into. Each epoch (training round) consists of dataset_size / batch_size training sets'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/tmp/training-output',
        help='Directory where Tensorflow checkpoints will be written'
    )
    parser.add_argument(
        '--restore',
        default=False,
        action='store_true',
        help='Restore from latest checkpoint in --output-dir'
    )
    parser.add_argument(
        '--video-dir',
        default='/tmp/training-videos',
        type=str,
        help='Directory where mp4s of each training epoch will be stored'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='learning_rate used by tf.train.RMSPropOptimizer'
    )
    parser.add_argument(
        '--rmsprop-decay',
        type=float,
        default=0.99,
        help='decay (gamma) used by tf.train.RMSPropOptimizer'
    )
    parser.add_argument(
        '--reward-decay',
        type=float,
        default=0.99,
        help='decay (gamma) used as a reward discount factor'
    )
    parser.add_argument(
        '--move-penalty',
        type=float,
        default=0.01,
        help='additional penalty (loss function multipler) applied when actor is moved, which discourages super-human bursts of movement'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=200
    )
    parser.add_argument(
        '--render',
        type=bool,
        default=True,
        help='Render gameplay visually (and record to --video-dir'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(logevel=args.loglevel)
    # save all checkpoints
    args.max_to_keep = args.n_epoch // args.save_checkpoint_steps

    main(args)
