import datetime
import csv

from os.path import splitext
from os import listdir

import pytz
from dateutil import parser
from influxdb import InfluxDBClient
from environments.market import Market, Candlestick

class MarketDb(object):

    @staticmethod
    def create_db(config):
        if config["engine"] == "test":
            return MarketTestDb(config)
        if config["engine"] == "text":
            return MarketTextDb(config)
        else:
            return MarketInfluxDb(config)


class MarketTestDb(object):

    def __init__(self, config):
        self.kline_num = config["minutes"]
        self.base = config["base"]

    def load(self, _trading_day):
        start_time_str = "2017-02-02T15:00:00Z"
        start_time = parser.parse(start_time_str)
        start_time = start_time.replace(tzinfo = pytz.timezone('Asia/Shanghai'))
        delta = datetime.timedelta(minutes = self.kline_num)
        end_time = start_time + delta

        klines = []
        markets = []
        for i in range(0, self.kline_num):
            kline_time = start_time + datetime.timedelta(minutes = i)
            kline_time_str = kline_time.strftime("%Y-%m-%d %H:%M:%S")
            kline_data = dict({"first": self.base, "last" : self.base, "min" : self.base, "max" : self.base, "qty" : 1, "time" : kline_time_str})
            kline = Candlestick(kline_data)
            klines.append(kline)
            market_data = dict({"last_price": self.base, "ask_price" : self.base, "ask_vol": 1, "bid_price": self.base, "bid_vol": 1, "qty": i, "time": kline_time_str})
            market = Market(market_data)
            markets.append(market)
        return markets, klines, start_time, end_time


class MarketTextDb(object):

    def __init__(self, config):
        self.directory = config["data"]
        files = listdir(self.directory)
        market_trading_days = []
        kline_trading_days = []
        self.trading_days = dict()
        for file in files:
            type_with_trading_day = MarketTextDb.type_with_trading_day(file)
            if type_with_trading_day:
                (file_type, file_trading_day) = type_with_trading_day
                if file_type == "market":
                    market_trading_days.append(file_trading_day)
                else:
                    kline_trading_days.append(file_trading_day)
        for trading_day in kline_trading_days:
            if trading_day in market_trading_days:
                self.trading_days[trading_day] = (self.kline_file(trading_day), self.market_file(trading_day))
            else:
                self.trading_days[trading_day] = (self.kline_file(trading_day), None)

    @staticmethod
    def type_with_trading_day(filename):
        root,ext = splitext(filename)
        if ext == ".csv":
            type_with_trading_day = root.split("_")
            length = len(type_with_trading_day)
            if length == 2:
                file_type = type_with_trading_day[0]
                trading_day = type_with_trading_day[1]
                if file_type == "market":
                    return file_type, trading_day
                if file_type == "kline":
                    return file_type, trading_day
                return None
            else:
                return None
        else:
            return None

    def get_trading_days(self):
        return list(self.trading_days.keys())

    def market_file(self, trading_day):
        return "{}/market_{}.csv".format(self.directory, trading_day)

    def kline_file(self, trading_day):
        return "{}/kline_{}.csv".format(self.directory, trading_day)

    def load(self, trading_day):
        files = self.trading_days[trading_day]
        print(files)
        candlesticks = []
        markets = []
        if files:
            (kline_file, market_file) = files
            with open(kline_file) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    candlestick = Candlestick(row)
                    candlesticks.append(candlestick)
            if market_file:
                with open(market_file) as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        market = Market(row)
                        markets.append(market)
        start_time = candlesticks[0].time
        end_time = candlesticks[len(candlesticks) - 1].time
        return markets, candlesticks, start_time, end_time

class MarketInfluxDb(object):

    def __init__(self, config):
        host = config["host"]
        port = config["port"]
        username = config["username"]
        password = config["password"]
        database = config["database"]
        self.client = InfluxDBClient(host=host, port= port, username=username, password=password, database=database)
        self.markets = []
        self.klines = []
        self.start_time = 0
        self.end_time = 0

    def load(self, _trading_day):
        start_time_str = "2017-02-02T15:00:00Z"
        end_time_str = "2017-02-02T16:00:00Z"
        start_time = parser.parse(start_time_str)
        start_time = start_time.replace(tzinfo = pytz.timezone('Asia/Shanghai'))
        end_time = parser.parse(end_time_str)
        end_time = end_time.replace(tzinfo = pytz.timezone('Asia/Shanghai'))
        sql_market = "select last_price, qty, ask_price, ask_vol, bid_price, bid_vol from markets where product_code = 'CL' and yearmonth = '1703' and time >= '%s' AND time < '%s'" % (start_time_str, end_time_str)
        sql_kline = "select max(last_price), min(last_price), first(last_price), last(last_price), last(qty) - first(qty) as qty from markets where product_code = 'CL' and yearmonth = '1703' and time >= '%s' AND time < '%s' group by time(1m)" % (start_time_str, end_time_str)
        rs_markets = self.client.query(sql_market)
        rs_klines = self.client.query(sql_kline)
        markets = list(map(Market, list(rs_markets.get_points())))
        klines = list(map(Candlestick, list(rs_klines.get_points())))
        return markets, klines, start_time, end_time
