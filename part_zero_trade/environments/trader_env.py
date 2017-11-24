import datetime
import numpy
from gym.spaces import Discrete
from environments.market import Market, Candlestick
from environments.market_db import MarketDb
from environments.trader import Trader

class TraderEnv(object):

    def __init__(self, config):
        self.market_db = MarketDb(config["database"])
        self.markets = []
        self.candlesticks = []
        self.offset = 0
        self.candlestick_num = 5
        self.start_time = 0
        self.end_time = 0
        self.max_offset = 0
        self.trading_day = 0
        self.trader = Trader(config["trader"])
        self.trader_config = config["trader"]
        self.markets, self.candlesticks, self.start_time, self.end_time = self.market_db.load(self.trading_day)

    def reset(self):
        self.offset = 0
        self.trader = Trader(self.trader_config)
        self.max_offset = len(self.markets)
        self.entry_num = 0
        self.exit_num = 0
        check_time = self.start_time + datetime.timedelta(minutes = self.candlestick_num)
        while True:
            market = self.markets[self.offset]
            if market.time < check_time:
                self.offset += 1
            else:
                break
        state = self.generate_state()
        return state

    def max_tick(self):
        return len(self.markets)

    def render(self):
        return 0

    def get_action_meanings(self):
        return "0: noop, 1: buy, 2: sell"

    @property
    def action_space(self):
        return Discrete(3)

    @property
    def observation_space(self):
        return Discrete(Trader.state_space_num() + Candlestick.state_space_num() * self.candlestick_num + Market.state_space_num())

    def generate_state(self):
        market = self.markets[self.offset]
        if market:
            candlestick_offset = self.minites_offset(market)
            state = self.trader.state()
            state += market.state()
            if candlestick_offset > self.candlestick_num:
                candlesticks = self.candlesticks[candlestick_offset -5:candlestick_offset]
                for candlestick in candlesticks:
                    state += candlestick.state()
            else:
                state += [0] * Candlestick.state_space_num() * self.candlestick_num
            return numpy.array([state])
        else:
            raise IndexError()

    def minites_offset(self, market):
            start_time = self.start_time
            current_time = market.time
            delta = datetime.timedelta(seconds = current_time.second, microseconds = current_time.microsecond)
            current_start_time = current_time - delta
            candlestick_delta = current_start_time - start_time
            return int(candlestick_delta.total_seconds()) // 60

    def step(self, action):
        reward = 0
        finished = False
        market = self.markets[self.offset]

        if self.offset + 1 < self.max_offset:
            if action == 1:
                reward = 0
                price = 0
            else:
                if action == 2:
                    direction = 1
                else:
                    direction = -1
                market = self.markets[self.offset]
                if direction == 1:
                    price = int(market.bid_price * 100)
                else:
                    price = int(market.ask_price * 100)
                if price != 0:
                    reward = self.trader.trade(direction, price, 1)
                else:
                    reward = 0
            self.offset += 1
        else:
            finished = True
            market = self.markets[self.max_offset - 1]
            price = int(market.last_price * 100)
            reward = self.trader.trade(- self.trader.direction, price, self.trader.volume)
        state = self.generate_state()
        return state, reward, finished, (price, market.time)
            