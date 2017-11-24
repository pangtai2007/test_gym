class Trader(object):
    def __init__(self, config):
        self.commission_rate = config["commission_rate"]
        self.margin_rate = config["margin_rate"]
        self.total_capital = config["init_capital"]
        self.volume_multiple = config["volume_multiple"]
        self.init_capital = self.total_capital
        self.max_equity = self.total_capital
        self.total_equity = self.total_capital
        self.equity_limit = config["equity_limit"]
        self.stop_loss = config["stop_loss"]
        self.hold_equity_reward = config["hold_equity_reward"]
        self.contracts = []
        self.volume = 0
        self.direction = 0
        self.margin = 0
        self.win_by_exit = 0
        self.avg_cost = 0
        self.commission = 0

    def update_avg_cost(self):
        total_cost = 0
        for contract in self.contracts:
            total_cost += contract.price * contract.volume
        if self.volume >0:
            self.avg_cost = total_cost / self.volume
        else:
            self.avg_cost = 0

    def avaliable_volume(self, price):
        margin_per_volume = int(round(price * self.margin_rate * self.volume_multiple))
        commission_per_volume = int(round(price * self.commission_rate))
        return self.total_capital / (margin_per_volume + commission_per_volume)

    def equity(self, price):
        equity = 0
        for contract in self.contracts:
              equity += (price - contract.price) * self.direction * self.volume_multiple * contract.volume
        self.total_equity = self.total_capital + self.margin + equity
        if self.total_equity > self.max_equity:
            self.max_equity = self.total_equity
        return self.total_equity

    def hold(self, price):
        reward = 0
        if self.hold_equity_reward:
            for contract in self.contracts:
                reward += (price - contract.last_price) * self.direction * self.volume_multiple * contract.volume
                contract.last_price = price
        return reward


    def check_equity(self, price):
        equity = self.equity(price)
        return equity < (self.max_equity * self.equity_limit)

    def check_loss(self, price):
        exits = []
        for i,contract in reversed(list(enumerate(self.contracts))):
            if((contract.price - price) * self.direction) > self.stop_loss:
                exits.append((i, contract.volume))
        return exits

    def trade(self, direction, price, volume):
        if self.direction == direction:
            return self.entry_trade(price, volume)
        elif self.direction == 0:
            reward = self.entry_trade(price, volume)
            if self.volume == 0:
                self.direction = 0
            else:
                self.direction = direction
            return reward
        else:
            if self.volume > volume:
                exit_volume = volume
                entry_volume = 0
            else:
                exit_volume = self.volume
                entry_volume = volume - self.volume
            reward = self.exit_trade(price, exit_volume)
            entry_reward = 0
            if entry_volume > 0:
                entry_reward = self.entry_trade(price, entry_volume)
                if self.volume == 0:
                    self.direction = 0
                else:
                    self.direction = direction
            return reward + entry_reward

    def entry_trade(self, price, volume):
        commission = int(round(price * self.commission_rate)) * volume
        margin_per_volume = int(round(price * self.margin_rate * self.volume_multiple))
        margin = margin_per_volume * volume
        total_capital = self.total_capital - commission - margin
        if total_capital > 0:
            contract = Contract(price, volume, margin_per_volume)
            self.contracts.append(contract)
            self.total_capital = total_capital
            self.margin += margin
            self.volume += volume
            self.update_avg_cost()
            self.commission += commission
            reward = - commission
            return reward
        else:
            return 0

    def exit_trade(self, price, volume, offset = 0):
        contract = self.contracts[offset]
        if contract.volume > volume:
            trade_volume = volume
        else:
            self.contracts.pop(offset)
            trade_volume = contract.volume
        contract.volume = contract.volume - trade_volume
        rest_volume = volume - trade_volume
        margin = contract.margin_per_volume * trade_volume
        if self.hold_equity_reward:
            cost = contract.price
        else:
            cost = contract.last_price
        win_by_exit = (price - contract.price) * self.direction * self.volume_multiple * trade_volume
        hold_win_by_exit = (price - cost) * self.direction * self.volume_multiple * trade_volume
        commission = int(round(price * self.commission_rate)) * volume
        self.total_capital += win_by_exit + margin -  commission
        self.volume -= trade_volume
        self.margin -= margin
        self.commission += commission
        self.update_avg_cost()
        self.win_by_exit += win_by_exit
        reward = hold_win_by_exit - commission
        if self.volume == 0:
            self.direction = 0
        if rest_volume > 0:
            return reward + self.exit_trade(price, rest_volume, 0)
        else:
            return reward

    @staticmethod
    def state_space_num():
        return 4

    def state(self, price):
        if price == 0:
            return [0, 0, 0, 0]
        else:
            equity = self.equity(price)
            if self.volume == 0:
                costp = 0
            else:
                costp = (self.avg_cost - price) / price * 100
            return [self.volume, self.direction, (equity - self.init_capital) / self.init_capital * 100, costp]


class Contract(object):

    def __init__(self, price, volume, margin_per_volume):
        self.price = price
        self.volume = volume
        self.margin_per_volume = margin_per_volume
        self.last_price = price

class Config(object):

    def __init__(self, margin_rate, commission_rate, init_capital, volume_multiple):
        self.margin_rate = margin_rate
        self.commission_rate = commission_rate
        self.init_capital = init_capital
        self.volume_multiple = volume_multiple
