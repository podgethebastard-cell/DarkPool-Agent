

import pandas as pd
import numpy as np
import datetime
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# ==========================================
# 1. Configuration & Constants
# ==========================================
TRANSACTION_COST = 0.001  # 0.1% per trade


@dataclass
class Trade:
    timestamp: datetime.datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    commission: float
    pnl: Optional[float] = None
    entry_price: Optional[float] = None


@dataclass
class Position:
    symbol: str
    quantity: float
    average_entry_price: float


# ==========================================
# 2. Mock Data Generator
# ==========================================
def generate_mock_market_data(num_points: int = 500, base_price: float = 100.0, 
                              volatility: float = 0.002, trend: float = 0.0001) -> pd.DataFrame:
    """
    Generates synthetic price data (OHLC) using a random walk.
    Returns a pandas DataFrame.
    """
    dates = pd.date_range(start=datetime.datetime.now(), periods=num_points, freq='1min')
    prices = []
    current_price = base_price

    for _ in range(num_points):
        # Simulate a random walk with configurable trend and volatility
        change_pct = np.random.normal(loc=trend, scale=volatility)
        current_price *= (1 + change_pct)
        
        # Generate OHLC roughly around the current price
        open_p = current_price
        high_p = open_p * (1 + abs(np.random.normal(0, 0.001)))
        low_p = open_p * (1 - abs(np.random.normal(0, 0.001)))
        close_p = (open_p + high_p + low_p) / 3
        
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
        end_idx = self._current_index
        start_idx = max(0, end_idx - window_size)
        return self._data.iloc[0:end_idx]
    
    def reset(self):
        """Reset cursor to beginning."""
        self._current_index = 0


# ==========================================
# 4. Strategy Engine - Multiple Strategies
# ==========================================
class BaseStrategy:
    """Base class for all trading strategies."""
    def generate_signal(self, history: pd.DataFrame) -> str:
        raise NotImplementedError


class MovingAverageCrossStrategy(BaseStrategy):
    """
    Simple Golden Cross Strategy.
    Buy when SMA_Short > SMA_Long.
    Sell when SMA_Short < SMA_Long.
    """
    def __init__(self, short_window: int, long_window: int):
        self.short_window = short_window
        self.long_window = long_window
        self.name = f"MA Cross ({short_window}/{long_window})"

    def generate_signal(self, history: pd.DataFrame) -> str:
        if len(history) < self.long_window:
            return 'HOLD'

        closes = history['close']
        short_ma = closes.rolling(window=self.short_window).mean()
        long_ma = closes.rolling(window=self.long_window).mean()

        if len(short_ma) < 2:
            return 'HOLD'

        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]

        # Golden Cross (Bullish)
        if prev_short <= prev_long and current_short > current_long:
            return 'BUY'
        
        # Death Cross (Bearish)
        elif prev_short >= prev_long and current_short < current_long:
            return 'SELL'
        
        return 'HOLD'


class RSIStrategy(BaseStrategy):
    """
    RSI-based strategy.
    Buy when RSI < oversold threshold, Sell when RSI > overbought threshold.
    """
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"RSI ({period}, {oversold}/{overbought})"

    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signal(self, history: pd.DataFrame) -> str:
        if len(history) < self.period + 1:
            return 'HOLD'

        closes = history['close']
        rsi = self.calculate_rsi(closes, self.period)
        
        if len(rsi) < 2:
            return 'HOLD'

        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        # Buy when RSI crosses above oversold
        if prev_rsi <= self.oversold and current_rsi > self.oversold:
            return 'BUY'
        
        # Sell when RSI crosses below overbought
        elif prev_rsi >= self.overbought and current_rsi < self.overbought:
            return 'SELL'
        
        return 'HOLD'


class MACDStrategy(BaseStrategy):
    """
    MACD crossover strategy.
    Buy when MACD crosses above signal line, Sell when below.
    """
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.name = f"MACD ({fast}/{slow}/{signal})"

    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = prices.ewm(span=self.fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.signal, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    def generate_signal(self, history: pd.DataFrame) -> str:
        if len(history) < self.slow + self.signal:
            return 'HOLD'

        closes = history['close']
        macd, signal, _ = self.calculate_macd(closes)

        if len(macd) < 2:
            return 'HOLD'

        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        prev_macd = macd.iloc[-2]
        prev_signal = signal.iloc[-2]

        # Buy when MACD crosses above signal
        if prev_macd <= prev_signal and current_macd > current_signal:
            return 'BUY'
        
        # Sell when MACD crosses below signal
        elif prev_macd >= prev_signal and current_macd < current_signal:
            return 'SELL'
        
        return 'HOLD'


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands strategy.
    Buy when price touches lower band, Sell when touches upper band.
    """
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        self.name = f"Bollinger Bands ({period}, {std_dev}σ)"

    def calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = prices.rolling(window=self.period).mean()
        std = prices.rolling(window=self.period).std()
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)
        return upper_band, sma, lower_band

    def generate_signal(self, history: pd.DataFrame) -> str:
        if len(history) < self.period + 1:
            return 'HOLD'

        closes = history['close']
        upper, middle, lower = self.calculate_bollinger_bands(closes)

        if len(upper) < 2:
            return 'HOLD'

        current_price = closes.iloc[-1]
        prev_price = closes.iloc[-2]
        current_lower = lower.iloc[-1]
        current_upper = upper.iloc[-1]
        prev_lower = lower.iloc[-2]

        # Buy when price bounces off lower band
        if prev_price <= prev_lower and current_price > current_lower:
            return 'BUY'
        
        # Sell when price touches upper band
        elif current_price >= current_upper:
            return 'SELL'
        
        return 'HOLD'


# ==========================================
# 5. Execution & Portfolio Management
# ==========================================
class PortfolioManager:
    def __init__(self, initial_capital: float, transaction_cost: float = TRANSACTION_COST):
        self.cash = initial_capital
        self.positions: List[Position] = []
        self.trade_log: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime.datetime] = []
        self.transaction_cost = transaction_cost
        self.open_trades: Dict[str, Trade] = {}  # Track open trades for PnL

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
        
        if signal == 'BUY':
            if self.cash > 0:
                quantity = (self.cash * 0.99) / price
                cost = quantity * price
                commission = cost * self.transaction_cost
                
                if self.cash >= (cost + commission):
                    self.cash -= (cost + commission)
                    
                    if position:
                        total_qty = position.quantity + quantity
                        avg_price = ((position.quantity * position.average_entry_price) + cost) / total_qty
                        position.quantity = total_qty
                        position.average_entry_price = avg_price
                    else:
                        new_pos = Position(symbol, quantity, price)
                        self.positions.append(new_pos)
                    
                    trade = Trade(timestamp, symbol, 'BUY', price, quantity, commission)
                    self.trade_log.append(trade)
                    self.open_trades[symbol] = trade

        elif signal == 'SELL':
            if position and position.quantity > 0:
                quantity = position.quantity
                revenue = quantity * price
                commission = revenue * self.transaction_cost
                
                self.cash += (revenue - commission)
                
                # Calculate PnL
                entry_trade = self.open_trades.get(symbol)
                pnl = None
                if entry_trade:
                    pnl = (price - entry_trade.price) * quantity - commission - entry_trade.commission
                    entry_trade.pnl = pnl
                    entry_trade.entry_price = entry_trade.price
                    del self.open_trades[symbol]
                else:
                    # Calculate PnL from position average entry price
                    pnl = (price - position.average_entry_price) * quantity - commission
                
                self.positions.remove(position)
                
                trade = Trade(timestamp, symbol, 'SELL', price, quantity, commission, 
                            pnl=pnl,
                            entry_price=position.average_entry_price)
                self.trade_log.append(trade)

    def update_equity(self, current_price: float, timestamp: datetime.datetime):
        position_value = sum([p.quantity * current_price for p in self.positions])
        total_equity = self.cash + position_value
        self.equity_curve.append(total_equity)
        self.timestamps.append(timestamp)
        return total_equity


# ==========================================
# 6. Performance Analytics
# ==========================================
class PerformanceAnalyzer:
    @staticmethod
    def calculate_metrics(portfolio: PortfolioManager, initial_capital: float) -> Dict:
        if not portfolio.equity_curve:
            return {}
        
        equity_series = pd.Series(portfolio.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        final_equity = portfolio.equity_curve[-1]
        total_return = ((final_equity - initial_capital) / initial_capital) * 100
        
        # Calculate win rate
        buy_trades = [t for t in portfolio.trade_log if t.side == 'BUY']
        sell_trades = [t for t in portfolio.trade_log if t.side == 'SELL']
        
        completed_trades = []
        for sell_trade in sell_trades:
            # Find corresponding buy trade
            for buy_trade in buy_trades:
                if buy_trade.symbol == sell_trade.symbol and buy_trade.timestamp < sell_trade.timestamp:
                    pnl = (sell_trade.price - buy_trade.price) * sell_trade.quantity - sell_trade.commission - buy_trade.commission
                    completed_trades.append({
                        'entry': buy_trade.price,
                        'exit': sell_trade.price,
                        'pnl': pnl,
                        'return_pct': ((sell_trade.price - buy_trade.price) / buy_trade.price) * 100
                    })
                    break
        
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        
        if completed_trades:
            winning_trades = [t for t in completed_trades if t['pnl'] > 0]
            losing_trades = [t for t in completed_trades if t['pnl'] < 0]
            
            win_rate = (len(winning_trades) / len(completed_trades)) * 100 if completed_trades else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            total_wins = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
            total_losses = abs(sum([t['pnl'] for t in losing_trades])) if losing_trades else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Risk metrics
        if len(returns) > 0:
            sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            max_drawdown = ((equity_series / equity_series.expanding().max()) - 1).min() * 100
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'final_equity': final_equity,
            'total_trades': len(portfolio.trade_log),
            'completed_trades': len(completed_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_series,
            'returns': returns
        }


# ==========================================
# 7. Main Agent Orchestrator
# ==========================================
class TradingAgent:
    def __init__(self, strategy, portfolio, data_cursor):
        self.strategy = strategy
        self.portfolio = portfolio
        self.cursor = data_cursor

    def run(self, progress_bar=None):
        while self.cursor.has_next():
            tick = self.cursor.next_tick()
            if tick is None:
                break

            history = self.cursor.get_history(window_size=200)
            signal = self.strategy.generate_signal(history)
            
            if signal != 'HOLD':
                self.portfolio.execute_trade(signal, tick)
            
            self.portfolio.update_equity(tick['close'], tick['timestamp'])
            
            if progress_bar:
                # FIX: Clamp progress value to maximum 1.0 to prevent StreamlitAPIException
                progress_value = self.cursor._current_index / self.cursor._max_index
                progress_bar.progress(min(progress_value, 1.0))


# ==========================================
# 8. Streamlit App
# ==========================================
def main():
    st.set_page_config(
        page_title="Titan Trading System",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Titan Trading System - Advanced Backtesting Platform")
    st.markdown("---")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Strategy Selection
        strategy_type = st.selectbox(
            "Trading Strategy",
            ["Moving Average Cross", "RSI", "MACD", "Bollinger Bands"]
        )
        
        # Strategy Parameters
        st.subheader("Strategy Parameters")
        
        if strategy_type == "Moving Average Cross":
            short_window = st.slider("Short MA Period", 5, 50, 20)
            long_window = st.slider("Long MA Period", 20, 200, 50)
            strategy = MovingAverageCrossStrategy(short_window, long_window)
        
        elif strategy_type == "RSI":
            rsi_period = st.slider("RSI Period", 5, 30, 14)
            oversold = st.slider("Oversold Level", 10, 40, 30)
            overbought = st.slider("Overbought Level", 60, 90, 70)
            strategy = RSIStrategy(rsi_period, oversold, overbought)
        
        elif strategy_type == "MACD":
            fast = st.slider("Fast EMA", 5, 20, 12)
            slow = st.slider("Slow EMA", 20, 50, 26)
            signal = st.slider("Signal Period", 5, 15, 9)
            strategy = MACDStrategy(fast, slow, signal)
        
        elif strategy_type == "Bollinger Bands":
            bb_period = st.slider("BB Period", 10, 50, 20)
            std_dev = st.slider("Standard Deviation", 1.0, 3.0, 2.0)
            strategy = BollingerBandsStrategy(bb_period, std_dev)
        
        st.markdown("---")
        
        # Market Data Parameters
        st.subheader("Market Data")
        data_points = st.slider("Number of Data Points", 100, 2000, 500)
        base_price = st.number_input("Base Price", 50.0, 500.0, 100.0)
        volatility = st.slider("Volatility", 0.0001, 0.01, 0.002, 0.0001)
        trend = st.slider("Trend", -0.001, 0.001, 0.0001, 0.0001)
        
        st.markdown("---")
        
        # Portfolio Parameters
        st.subheader("Portfolio")
        initial_capital = st.number_input("Initial Capital ($)", 1000.0, 100000.0, 10000.0)
        transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1) / 100
        
        st.markdown("---")
        
        # Run Backtest Button
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            st.session_state.run_backtest = True
    
    # Main Content Area
    if st.session_state.get('run_backtest', False):
        with st.spinner("Running backtest..."):
            # Generate data
            market_data = generate_mock_market_data(data_points, base_price, volatility, trend)
            
            # Initialize components
            data_cursor = MarketDataCursor(market_data)
            portfolio = PortfolioManager(initial_capital, transaction_cost)
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run agent
            agent = TradingAgent(strategy, portfolio, data_cursor)
            agent.run(progress_bar)
            
            progress_bar.empty()
            status_text.empty()
            
            # Calculate metrics
            analyzer = PerformanceAnalyzer()
            metrics = analyzer.calculate_metrics(portfolio, initial_capital)
            
            st.session_state.backtest_results = {
                'market_data': market_data,
                'portfolio': portfolio,
                'strategy': strategy,
                'metrics': metrics
            }
            
            st.session_state.run_backtest = False
    
    # Display Results
    if 'backtest_results' in st.session_state:
        results = st.session_state.backtest_results
        portfolio = results['portfolio']
        metrics = results['metrics']
        market_data = results['market_data']
        strategy = results['strategy']
        
        # Key Metrics Row
        st.subheader("📊 Performance Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%",
                     delta=f"${metrics.get('final_equity', 0) - initial_capital:.2f}")
        
        with col2:
            st.metric("Final Equity", f"${metrics.get('final_equity', 0):.2f}",
                     delta=f"{metrics.get('total_return', 0):.2f}%")
        
        with col3:
            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%",
                     delta=f"{metrics.get('completed_trades', 0)} trades")
        
        with col4:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        
        with col5:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
        
        st.markdown("---")
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Equity Curve")
            if portfolio.equity_curve:
                equity_df = pd.DataFrame({
                    'Timestamp': portfolio.timestamps,
                    'Equity': portfolio.equity_curve
                })
                fig = px.line(equity_df, x='Timestamp', y='Equity', 
                            title=f"Portfolio Equity Over Time - {strategy.name}")
                fig.add_hline(y=initial_capital, line_dash="dash", 
                            line_color="gray", annotation_text="Initial Capital")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💰 Returns Distribution")
            if metrics.get('returns') is not None and len(metrics['returns']) > 0:
                returns_df = pd.DataFrame({'Returns': metrics['returns']})
                fig = px.histogram(returns_df, x='Returns', nbins=50,
                                 title="Distribution of Returns")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Charts Row 2 - Price Chart with Trades
        st.subheader("📊 Price Chart with Trade Signals")
        fig = make_subplots(rows=2, cols=1, 
                          shared_xaxes=True,
                          vertical_spacing=0.03,
                          subplot_titles=('Price & Indicators', 'Volume'),
                          row_heights=[0.7, 0.3])
        
        # Price line
        fig.add_trace(
            go.Scatter(x=market_data['timestamp'], y=market_data['close'],
                      name='Close Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add strategy indicators
        if isinstance(strategy, MovingAverageCrossStrategy):
            closes = market_data['close']
            short_ma = closes.rolling(window=strategy.short_window).mean()
            long_ma = closes.rolling(window=strategy.long_window).mean()
            fig.add_trace(
                go.Scatter(x=market_data['timestamp'], y=short_ma,
                          name=f'SMA {strategy.short_window}', line=dict(color='orange', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=market_data['timestamp'], y=long_ma,
                          name=f'SMA {strategy.long_window}', line=dict(color='red', width=1)),
                row=1, col=1
            )
        
        # Add buy/sell markers
        buy_trades = [t for t in portfolio.trade_log if t.side == 'BUY']
        sell_trades = [t for t in portfolio.trade_log if t.side == 'SELL']
        
        if buy_trades:
            buy_times = [t.timestamp for t in buy_trades]
            buy_prices = [t.price for t in buy_trades]
            fig.add_trace(
                go.Scatter(x=buy_times, y=buy_prices, mode='markers',
                          name='BUY', marker=dict(symbol='triangle-up', size=10, color='green')),
                row=1, col=1
            )
        
        if sell_trades:
            sell_times = [t.timestamp for t in sell_trades]
            sell_prices = [t.price for t in sell_trades]
            fig.add_trace(
                go.Scatter(x=sell_times, y=sell_prices, mode='markers',
                          name='SELL', marker=dict(symbol='triangle-down', size=10, color='red')),
                row=1, col=1
            )
        
        # Volume
        fig.add_trace(
            go.Bar(x=market_data['timestamp'], y=market_data['volume'],
                  name='Volume', marker_color='lightblue'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Metrics
        st.subheader("📋 Detailed Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Trade Statistics**")
            st.json({
                "Total Trades": metrics.get('total_trades', 0),
                "Completed Trades": metrics.get('completed_trades', 0),
                "Win Rate": f"{metrics.get('win_rate', 0):.2f}%",
                "Average Win": f"${metrics.get('avg_win', 0):.2f}",
                "Average Loss": f"${metrics.get('avg_loss', 0):.2f}",
                "Profit Factor": f"{metrics.get('profit_factor', 0):.2f}"
            })
        
        with col2:
            st.markdown("**Risk Metrics**")
            st.json({
                "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.2f}",
                "Maximum Drawdown": f"{metrics.get('max_drawdown', 0):.2f}%",
                "Total Return": f"{metrics.get('total_return', 0):.2f}%",
                "Final Equity": f"${metrics.get('final_equity', 0):.2f}",
                "Cash Remaining": f"${portfolio.cash:.2f}",
                "Open Positions": len(portfolio.positions)
            })
        
        # Trade History Table
        st.subheader("📜 Trade History")
        if portfolio.trade_log:
            trades_df = pd.DataFrame([{
                'Timestamp': t.timestamp,
                'Side': t.side,
                'Symbol': t.symbol,
                'Price': f"${t.price:.2f}",
                'Quantity': f"{t.quantity:.4f}",
                'Commission': f"${t.commission:.2f}",
                'PnL': f"${t.pnl:.2f}" if t.pnl else "N/A"
            } for t in portfolio.trade_log])
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades executed during this backtest.")
    
    else:
        # Welcome Screen
        st.info("👈 Configure your strategy and parameters in the sidebar, then click 'Run Backtest' to start!")
        
        st.markdown("""
        ### 🎯 Features
        
        - **Multiple Trading Strategies**: MA Cross, RSI, MACD, Bollinger Bands
        - **Interactive Parameter Tuning**: Adjust strategy parameters in real-time
        - **Comprehensive Analytics**: Performance metrics, risk analysis, and trade statistics
        - **Visual Charts**: Equity curves, price charts with signals, and returns distribution
        - **Detailed Trade Log**: Complete history of all executed trades
        
        ### 📚 How to Use
        
        1. Select a trading strategy from the sidebar
        2. Adjust strategy parameters to your preference
        3. Configure market data parameters (volatility, trend, etc.)
        4. Set your initial capital and transaction costs
        5. Click "Run Backtest" to execute the simulation
        6. Analyze the results and optimize your strategy!
        """)


if __name__ == "__main__":
    main()
