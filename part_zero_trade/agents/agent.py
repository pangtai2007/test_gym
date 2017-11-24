import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import pdb
from logging import getLogger
import objgraph

from .history import History
from .experience import Experience

logger = getLogger(__name__)

def get_time():
  return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())

class Agent(object):
  def __init__(self, sess, pred_network, env, stat, conf, target_network=None):
    self.sess = sess
    self.stat = stat

    self.ep_start = conf.ep_start
    self.ep_end = conf.ep_end
    self.history_length = conf.history_length
    self.t_ep_end = conf.t_ep_end
    self.t_learn_start = conf.t_learn_start
    self.t_train_freq = conf.t_train_freq
    self.t_target_q_update_freq = conf.t_target_q_update_freq
    self.env_name = conf.env_name

    self.discount_r = conf.discount_r
    self.min_r = conf.min_r
    self.max_r = conf.max_r
    self.min_delta = conf.min_delta
    self.max_delta = conf.max_delta
    self.max_grad_norm = conf.max_grad_norm
    self.observation_dims = conf.observation_dims

    self.learning_rate = conf.learning_rate
    self.learning_rate_minimum = conf.learning_rate_minimum
    self.learning_rate_decay = conf.learning_rate_decay
    self.learning_rate_decay_step = conf.learning_rate_decay_step

    # network
    self.double_q = conf.double_q
    self.pred_network = pred_network
    self.target_network = target_network
    self.target_network.create_copy_op(self.pred_network)
    self.learning_rate_op = 0

    self.tick = 0

    self.buy_num = 0
    self.sell_num = 0
    self.rand_num = 0
    self.clear_num = 0

    self.env = env
    self.history = History(
        conf.data_format, conf.batch_size, conf.history_length, conf.observation_dims)
    self.experience = Experience(
        conf.data_format, conf.batch_size,
        conf.history_length, conf.memory_size, conf.observation_dims)

  def train(self, t_max):
    tf.global_variables_initializer().run()

    self.stat.load_model()
    self.target_network.run_copy()

    start_t = self.stat.get_t()
    observation, reward, terminal = self.env.new_game()

    episodes = 0

    for _ in range(self.history_length):
      self.history.add(observation)
    
    total_reward = 0

    for self.tick in tqdm(range(start_t, t_max), ncols=70, initial=start_t):
      elta = (self.ep_end +
            max(0., (self.ep_start - self.ep_end)
                * (self.t_ep_end - max(0., self.tick - self.t_learn_start)) / self.t_ep_end))

      # 1. predict
      action = self.predict(self.history.get(), elta)
      # 2. act
      observation, reward, terminal, (_price, early_exit, fbuy_num, fsell_num, fclear_num) = self.env.step(action, True)
      # 3. observe
      theq, loss, is_update = self.observe(observation, reward, action, terminal)

      total_reward += reward
      if self.stat:
        self.stat.on_step(self.tick, action, reward, terminal,
                          elta, theq, loss, is_update, self.learning_rate_op)
      
      if terminal:
        print("buy: %d, sell: %d, clear: %d, fbuy: %d fsell: %d, fclear: %d, rand: %d r: %f, t: %d, q: %.4f, l: %.10f" % (self.buy_num, self.sell_num, self.clear_num, fbuy_num, fsell_num, fclear_num, self.rand_num, total_reward, terminal, np.mean(theq), loss))
        self.buy_num = 0
        self.sell_num = 0
        self.clear_num = 0
        self.rand_num = 0
        total_reward = 0
        episodes += 1
        if early_exit:
          self.env.continue_game()
        else:
          self.env.new_random_game()



  def play(self, test_ep, n_step=10000, n_episode=100):
    tf.global_variables_initializer().run()

    self.stat.load_model()
    self.target_network.run_copy()

    if not self.env.display:
      gym_dir = '/tmp/%s-%s' % (self.env_name, get_time())
      self.env.env.monitor.start(gym_dir)

    best_reward, _best_idx, best_count = 0, 0, 0
    trading_days = self.env.env.trading_days
    
    for trading_day in list(trading_days):
      observation, reward, terminal = self.env.new_trading_day_game(trading_day)
      current_reward = 0

      for _ in range(self.history_length):
        self.history.add(observation)

      for self.tick in tqdm(range(test_ep), ncols=70):
        # 1. predict
        action = self.predict(self.history.get(), 0)
        # 2. act
        observation, reward, terminal, (_price, early_exit, _fbuy_num, _fsell_num, _fclear_num) = self.env.step(action, False)
        self.history.add(observation)

        current_reward += reward

        if terminal:
          if early_exit:
            self.env.continue_game()
          else:
            break

      if current_reward > best_reward:
        best_reward = current_reward
        best_count = 0
      elif current_reward == best_reward:
        best_count += 1

    if not self.env.display:
      self.env.env.monitor.close()

  def predict(self, s_t, elta):
    if random.random() < elta:
      action = random.randrange(self.env.action_size)
      self.rand_num +=1
    else:
      action = self.pred_network.calc_actions([s_t])[0]
      if action == 2:
        self.buy_num +=1
      if action == 0:
        self.sell_num +=1
      if action == 3:
        self.clear_num += 1
    return action

  def observe(self, observation, reward, action, terminal):
    raise NotImplementedError()

  def update_target_q_network(self):
    assert self.target_network != None
    self.target_network.run_copy()
