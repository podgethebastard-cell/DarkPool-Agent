import pandas as pd
import numpy as np
import datetime
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ==========================================
# 1. Configuration & Constants
# ==========================================
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001  # 0.1% per trade
SHORT_WINDOW = 20
LONG_WINDOW = 50
DATA_POINTS = 500  # Number of simulated candles


@dataclass
class Trade:
    timestamp: datetime.datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    commission: float
    pnl: float = 0.0  # Added field to track realized PnL per trade easily


@dataclass
class Position:
    symbol: str
    quantity: float
    average_entry_price: float


# ==========================================
# 2. Mock Data Generator
# ==========================================
def generate_mock_market_data(num_points: int = 500) -> pd.DataFrame:
    """
    Generates synthetic price data (OHLC) using a random walk.
    Returns a pandas DataFrame.
    """
    base_price = 100.0
    dates = pd.date_range(start=datetime.datetime.now(), periods=num_points, freq='1min')
    prices = []
    current_price = base_price

    for _ in range(num_points):
        # Simulate a random walk
        change_pct = np.random.normal(loc=0.0001, scale=0.002)
        current_price *= (1 + change_pct)
        
        # Generate OHLC roughly around the current price
        open_p = current_price
        high_p = open_p * (1 + abs(np.random.normal(0, 0.001)))
        low_p = open_p * (1 - abs(np.random.normal(0, 0.001)))
        close_p = (open_p + high_p + low_p) / 3 # Rough approximation
        
        prices.append({
            'timestamp': dates[_],
            'symbol': 'BTC-USD',
            'open': open_p,
            'high': high_p,
            'low': low_p,
            'close': close_p,
            'volume': np.random.randint(10, 100)
        })
    
    return pd.DataFrame(prices)


# ==========================================
# 3. The Cursor (Data Handler)
# ==========================================
class MarketDataCursor:
    """
    Iterates through a static dataset as if it were a live stream.
    This allows the agent to 'see' data only up to the current moment.
    """
    def __init__(self, historical_data: pd.DataFrame):
        self._data = historical_data.sort_values('timestamp').reset_index(drop=True)
        self._current_index = 0
        self._max_index = len(self._data) - 1

    def has_next(self) -> bool:
        return self._current_index <= self._max_index

    def next_tick(self) -> Optional[pd.Series]:
        """Returns the next row of data and advances the cursor."""
        if not self.has_next():
            return None
        
        row = self._data.iloc[self._current_index]
        self._current_index += 1
        return row

    def get_history(self, window_size: int = 100) -> pd.DataFrame:
        """
        Returns data from the beginning up to the current cursor position.
        Used by the strategy to calculate indicators (SMA, RSI, etc.).
        """
        # Ensure we don't look ahead
        end_idx = self._current_index
        start_idx = max(0, end_idx - window_size)
        
        # [FIXED] Correctly using start_idx to slice the dataframe
        return self._data.iloc[start_idx:end_idx]


# ==========================================
# 4. Strategy Engine
# ==========================================
class MovingAverageCrossStrategy:
    """
    Simple Golden Cross Strategy.
    Buy when SMA_Short > SMA_Long.
    Sell when SMA_Short < SMA_Long.
    """
    def __init__(self, short_window: int, long_window: int):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(self, history: pd.DataFrame) -> str:
        """
        Analyzes history and returns 'BUY', 'SELL', or 'HOLD'.
        """
        if len(history) < self.long_window:
            return 'HOLD'

        # Calculate Indicators
        # Note: In a real optimize scenario, we would update incremental values 
        # instead of recalculating the whole series every tick.
        closes = history['close']
        short_ma = closes.rolling(window=self.short_window).mean()
        long_ma = closes.rolling(window=self.long_window).mean()

        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]

        # Check for Crossover
        # Golden Cross (Bullish)
        if prev_short <= prev_long and current_short > current_long:
            return 'BUY'
        
        # Death Cross (Bearish)
        elif prev_short >= prev_long and current_short < current_long:
            return 'SELL'
        
        return 'HOLD'


# ==========================================
# 5. Execution & Portfolio Management
# ==========================================
class PortfolioManager:
    def __init__(self, initial_capital: float):
        self.cash = initial_capital
        self.positions: List[Position] = []
        self.trade_log: List[Trade] = []
        self.equity_curve: List[float] = []

    def get_position(self, symbol: str) -> Optional[Position]:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None

    def execute_trade(self, signal: str, market_data: pd.Series):
        symbol = market_data['symbol']
        price = market_data['close']
        timestamp = market_data['timestamp']
        
        position = self.get_position(symbol)
        
        # Simple Sizing: Use 100% of cash for buys, sell 100% of position for sells
        if signal == 'BUY':
            if self.cash > 0:
                quantity = (self.cash * 0.99) / price  # Leave buffer for comms
                cost = quantity * price
                commission = cost * TRANSACTION_COST
                
                if self.cash >= (cost + commission):
                    self.cash -= (cost + commission)
                    
                    if position:
                        # Average up (simplified)
                        total_qty = position.quantity + quantity
                        avg_price = ((position.quantity * position.average_entry_price) + cost) / total_qty
                        position.quantity = total_qty
                        position.average_entry_price = avg_price
                    else:
                        new_pos = Position(symbol, quantity, price)
                        self.positions.append(new_pos)
                        
                    self.trade_log.append(Trade(timestamp, symbol, 'BUY', price, quantity, commission))
                    print(f"[{timestamp}] BUY  {symbol} @ {price:.2f} | Qty: {quantity:.4f}")

        elif signal == 'SELL':
            if position and position.quantity > 0:
                quantity = position.quantity
                revenue = quantity * price
                commission = revenue * TRANSACTION_COST
                
                self.cash += (revenue - commission)
                
                # Calculate PnL for this trade
                realized_pnl = (price - position.average_entry_price) * quantity
                
                self.positions.remove(position)
                
                self.trade_log.append(Trade(timestamp, symbol, 'SELL', price, quantity, commission, pnl=realized_pnl))
                print(f"[{timestamp}] SELL {symbol} @ {price:.2f} | PnL: {realized_pnl:.2f}")

    def update_equity(self, current_price: float):
        position_value = sum([p.quantity * current_price for p in self.positions])
        total_equity = self.cash + position_value
        self.equity_curve.append(total_equity)
        return total_equity


# ==========================================
# 6. Main Agent Orchestrator
# ==========================================
class TradingAgent:
    def __init__(self, strategy, portfolio, data_cursor):
        self.strategy = strategy
        self.portfolio = portfolio
        self.cursor = data_cursor

    def run(self):
        print("Starting Trading Agent Backtest...")
        print("-" * 50)
        
        while self.cursor.has_next():
            # 1. Get next market tick (Simulate live feed)
            tick = self.cursor.next_tick()
            if tick is None:
                break

            # 2. Get available history from cursor for strategy
            history = self.cursor.get_history(window_size=LONG_WINDOW + 5)
            
            # 3. Strategy Logic
            signal = self.strategy.generate_signal(history)
            
            # 4. Execution Logic
            if signal != 'HOLD':
                self.portfolio.execute_trade(signal, tick)
            
            # 5. Track Performance
            current_equity = self.portfolio.update_equity(tick['close'])

        self.generate_report()

    def generate_report(self):
        print("-" * 50)
        print("Backtest Complete.")
        print("-" * 50)
        
        final_equity = self.portfolio.equity_curve[-1]
        roi = ((final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        
        print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
        print(f"Final Equity:    ${final_equity:.2f}")
        print(f"Total Trades:    {len(self.portfolio.trade_log)}")
        print(f"Return (ROI):    {roi:.2f}%")
        
        # Calculate Win Rate
        if len(self.portfolio.trade_log) > 0:
            # Filter only for SELL trades (exits) to calculate win rate
            exits = [t for t in self.portfolio.trade_log if t.side == 'SELL']
            if exits:
                wins = len([t for t in exits if t.pnl > 0])
                win_rate = (wins / len(exits)) * 100
                print(f"Winning Trades:  {wins}")
                print(f"Losing Trades:   {len(exits) - wins}")
                print(f"Win Rate:        {win_rate:.2f}%")
            else:
                print("Win Rate:        N/A (No positions closed)")
        else:
            print("Win Rate:        N/A (No trades)")


# ==========================================
# 7. Execution Entry Point
# ==========================================
if __name__ == "__main__":
    # 1. Setup Data
    print("Generating Mock Data...")
    market_data = generate_mock_market_data(DATA_POINTS)
    
    # 2. Initialize Components
    # The 'Cursor' here is the key component requested
    data_cursor = MarketDataCursor(market_data)
    
    strategy = MovingAverageCrossStrategy(short_window=SHORT_WINDOW, long_window=LONG_WINDOW)
    portfolio = PortfolioManager(initial_capital=INITIAL_CAPITAL)
    
    # 3. Initialize Agent
    agent = TradingAgent(strategy, portfolio, data_cursor)
    
    # 4. Run
    agent.run()

