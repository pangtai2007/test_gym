import random
from environments.candlestick_trader_env import TraderEnv

class TraderEnvironment(object):
  def __init__(self, data_format, config):

    self.env = TraderEnv(config)
    self.action_size = self.env.action_space.n
    self.data_format = data_format
    self.last_trade_price = 0
    self.terminal = False
    self.last_action = 0
    self.last_reward = 0
    self.trade_time = 0
    self.buy_num = 0
    self.sell_num = 0
    self.clear_num = 0
    self.display = config["display"]

    if self.display:
      self.file = open(config["out_file"], "w+")
      self.env.file = self.file

  @property
  def observation_dims(self):
      return [1, self.env.observation_space.n]

  @property
  def max_tick(self):
    return self.env.max_tick()

  def new_game(self):
    self.buy_num = 0
    self.sell_num = 0

    screen = self.env.reset()
    return screen, 0, False

  def new_trading_day_game(self, trading_day):
    self.buy_num = 0
    self.sell_num = 0
    self.clear_num = 0
    screen = self.env.reset_trading_day(trading_day)
    return screen, 0, False

  def new_random_game(self):
    self.buy_num = 0
    self.sell_num = 0
    self.clear_num = 0
    trading_days = self.env.trading_days
    offset = random.randrange(len(trading_days))
    trading_day = trading_days[offset]
    screen = self.env.reset_trading_day(trading_day)
    return screen, 0, False

  def continue_game(self):
    self.buy_num = 0
    self.sell_num = 0

    screen = self.env.continue_game()
    return screen, 0, False

  def step(self, action, is_train):
    if action == 2:
      self.buy_num += 1
    if action == 0:
      self.sell_num += 1
    if action == 3:
      self.clear_num += 1

    state, reward, terminal, (price,time, early_exit) = self.env.step(action, is_train)
    self.last_action = action
    if action != 1:
      self.last_trade_price = price
      self.trade_time = time
      self.last_reward = reward
    self.terminal = terminal
    return state, reward, terminal, (price, early_exit, self.buy_num, self.sell_num, self.clear_num)
