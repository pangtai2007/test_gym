import datetime
import math
import numpy
from gym.spaces import Discrete
from environments.market_db import MarketDb
from environments.market import Candlestick
from environments.market import TradingDayData
from environments.trader import Trader


class TraderEnv(object):

    def __init__(self, config):
        self.market_db = MarketDb.create_db(config["database"])
        self.display = config["display"]
        self.candlesticks = []
        self.offset = 0
        self.market_offset = 0
        self.candlestick_num = 20
        self.start_time = 0
        self.end_time = 0
        self.max_offset = 0
        self.trading_day = 0
        self.file = None
        self.trader_config = config["trader"]
        self.trader = Trader(self.trader_config)
        self.volume_level = config["trader"]["volume_level"]
        self.trading_days = self.market_db.get_trading_days()
        self.datas = {}
        for trading_day in self.trading_days:
            markets, candlesticks, start_time, end_time = self.market_db.load(trading_day)
            self.datas[trading_day] = TradingDayData(markets, candlesticks, start_time, end_time)
            self.trading_day = trading_day
        self.reset_trading_day(self.trading_day)
        self.first_display = True

    def reset_trading_day(self, trading_day):
        self.trading_day = trading_day
        data = self.datas[trading_day]
        self.candlesticks = data.candlesticks
        self.start_time = data.start_time
        self.end_time = data.end_time
        return self.reset()

    def reset(self):
        self.trader = Trader(self.trader_config)
        self.max_offset = len(self.candlesticks)
        self.offset = self.candlestick_num
        self.market_offset = 0
        state = self.generate_state(0)
        return state

    def continue_game(self):
        self.trader.init_capital = self.trader.total_capital
        self.trader.max_equity = self.trader.total_capital
        self.trader.total_equity = self.trader.total_capital
        state = self.generate_state(self.offset)
        return state

    def render(self):
        return 0

    def max_tick(self):
        return len(self.candlesticks)

    def get_action_meanings(self):
        return "0: noop, 1: buy, 2: sell, 3: clear position"

    @property
    def action_space(self):
        return Discrete(4)

    @property
    def observation_space(self):
        return Discrete(Trader.state_space_num() + Candlestick.state_space_num() * self.candlestick_num + 1)

    def generate_state(self, price):
        candlestick_offset = self.offset
        state = self.trader.state(price)
        if self.settled():
            settle = 1
        else:
            settle = 0
        if candlestick_offset >= self.candlestick_num:
            candlesticks = self.candlesticks[candlestick_offset - self.candlestick_num:candlestick_offset]
        else:
            candlesticks = self.candlesticks[0:candlestick_offset]
            state += [0] * Candlestick.state_space_num() * (self.candlestick_num - candlestick_offset)
        for candlestick in candlesticks:
            state += candlestick.state()
        state += [settle]
        return numpy.array([state])

    def settled(self):
        return (self.offset + 1) >= self.max_offset - 3

    def minutes_offset(self, market):
        start_time = self.start_time
        current_time = market.time
        delta = datetime.timedelta(seconds=current_time.second, microseconds=current_time.microsecond)
        current_start_time = current_time - delta
        candlestick_delta = current_start_time - start_time
        return int(candlestick_delta.total_seconds()) // 60

    def step(self, action, is_train=True):
        reward = 0
        finished = False
        candlestick = self.candlesticks[self.offset]
        reward = self.trader.hold(candlestick.last) / self.trader.init_capital * 100
        self.offset += 1
        next_candlestick = self.candlesticks[self.offset]
        if self.trader.direction == 1:
            trade_price = next_candlestick.bid_price
        elif self.trader.direction == -1:
            trade_price = next_candlestick.ask_price
        else:
            trade_price = candlestick.last
        early_exit = False
        if self.settled():
            finished = True
            reward += self.trade(- self.trader.direction, trade_price, self.trader.volume, candlestick.time, True)
        else:
            if self.trader.check_equity(candlestick.last):
                reward += self.trade(- self.trader.direction, trade_price, self.trader.volume, candlestick.time, is_train)
                early_exit = True
                finished = True
            else:
                exits = self.trader.check_loss(candlestick.last)
                for (exit_offset, volume) in exits:
                    reward += self.trade(self.trader.direction, trade_price, volume, candlestick.time, False, exit_offset)
                if action == 1:
                    volume = 0
                else:
                    if action == 3:
                        direction = - self.trader.direction
                        volume = self.trader.volume
                        if volume == 0:
                            reward += - int(self.trader.commission_rate * candlestick.last) / self.trader.init_capital * 100
                    elif action == 2:
                        direction = 1
                        volume = self.trade_volume(direction, candlestick.last)
                    else:
                        direction = -1
                        volume = self.trade_volume(direction, candlestick.last)
                    if direction == 1:
                        trade_price = candlestick.ask_price
                    else:
                        trade_price = candlestick.bid_price
                    reward += self.trade(direction, trade_price, volume, candlestick.time, False)
        state = self.generate_state(candlestick.last)
        return state, reward, finished, (trade_price, candlestick.time, early_exit)

    def trade_volume(self, direction, price):
        avaliable = math.ceil(self.trader.avaliable_volume(price) - 0.1)
        current = self.trader.volume
        future = round((avaliable + current) * self.volume_level)
        if self.trader.direction == - direction:
            if future > current:
                future = current
        else:
            if future > avaliable:
                future = avaliable
        return future
    
    def hold_reward(self, candlestick):
        return (candlestick.last - candlestick.first) * self.trader.direction * self.trader.volume_multiple * self.trader.volume

    def trade(self, direction, price, volume, time, terminate, offset = None):
        reward = 0
        init_volume = self.trader.volume

        pre = self.trader.win_by_exit - self.trader.commission
        if offset:
            reward = self.trader.exit_trade(price, volume, offset)
            direction = - self.trader.direction
        else:
            for _ in range(volume):
                reward += self.trader.trade(direction, price, 1)

        if direction == self.trader.direction:
            volume = self.trader.volume - init_volume

        actual = self.trader.win_by_exit - self.trader.commission - pre


        if self.display:
            if self.first_display:
                self.file.write("time,direction,price,volume,reward,win_by_exit,commission,terminate\n")
                self.first_display = False
            if direction ==1 :
                action = "BULL"
            else:
                action = "BEAR"
            if (volume != 0) or terminate:
                self.file.write("%s,%s,%d,%d,%d,%d,%d,%d\n" % (time.isoformat(' '),action, price, volume, actual,self.trader.win_by_exit,self.trader.commission,terminate))
        return reward / self.trader.init_capital * 100
