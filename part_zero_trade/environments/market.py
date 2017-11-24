import datetime
from dateutil import parser
import pytz

class Market(object):

    def __init__(self, data):
        self.last_price = int(data["last"])
        self.qty = int(float(data["volume"]))
        self.code = data["symbol"]
        self.ask_price = int(data["ask_price"])
        self.ask_volume = int(data["ask_size"])
        self.bid_price = int(data["bid_price"])
        self.bid_volume = int(data["bid_size"])
        self.time = datetime.datetime.fromtimestamp(int(data["time"]) // 1000000000)
        #self.time = parser.parse(data["time"])
        timezone = data["timezone"].replace("GMT+", "Etc/GMT+")
        self.time = self.time.replace(tzinfo = pytz.timezone(timezone))

    @staticmethod
    def state_space_num():
        return 7

    def state(self):
        return [self.last_price, self.qty, self.ask_price, self.ask_volume, self.bid_price, self.bid_volume, self.time.timestamp()]

class Candlestick(object):

    def __init__(self, data):
        self.first = int(data["open"]) or 0
        self.last = int(data["close"]) or 0
        self.min = int(data["low"]) or 0
        self.max = int(data["high"]) or 0
        self.ask = 0
        self.bid = 0
        self.qty = int(float(data["volume"])) or 0
        self.code = data["symbol"]
        self.time = datetime.datetime.fromtimestamp(int(data["time"]) // 1000000000)
        self.trading_day = parser.parse(data["trading_day"])
        #self.time = parser.parse(data["time"])
        timezone = data["timezone"].replace("GMT+", "Etc/GMT+")
        self.time = self.time.replace(tzinfo = pytz.timezone(timezone))


    @staticmethod
    def state_space_num():
        return 3

    def state(self):
        return [self.percent(self.last), self.percent(self.min), self.percent(self.max)]

    def percent(self, item):
        return (item - self.first) / self.first * 100

class TradingDayData(object):

    def __init__(self, markets, candlesticks, start_time, end_time):
        self.candlesticks = candlesticks
        self.candlesticks_num = len(candlesticks)
        self.start_time = start_time
        self.end_time = end_time
        market_offset = 0
        for candlestick in candlesticks:
            while True:
                if market_offset < len(markets):
                    market = markets[market_offset]
                    if market.time < candlestick.time:
                        market_offset += 1
                    elif market.ask_price <= 0 or market.bid_price <= 0 or market.ask_volume <= 0 or market.bid_volume <= 0:
                        market_offset += 1
                    else:
                        candlestick.ask_price = market.ask_price
                        candlestick.bid_price = market.bid_price
                        break
                else:
                    candlestick.ask_price = 0
                    candlestick.bid_price = 0
                    break

