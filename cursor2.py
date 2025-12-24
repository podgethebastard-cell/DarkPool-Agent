
```python
import pandas as pd
import numpy as np
import datetime
import random
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Dict
from enum import Enum

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st
from google import genai

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

DEFAULT_MODEL = st.secrets.get("DEFAULT_GEMINI_MODEL", "gemini-3-pro-preview")

resp = client.models.generate_content(
    model=DEFAULT_MODEL,
    contents="Give me a market regime read for BTC on 1H and 4H.",
)
st.write(resp.text)


# ==========================================
# 1) Configuration & Constants
# ==========================================
DEFAULT_COMMISSION_RATE = 0.001  # 0.1%


# ==========================================
# 2) Core Data Models
# ==========================================
@dataclass
class Trade:
    timestamp: datetime.datetime
    symbol: str
    side: str  # BUY / SELL
    price: float
    quantity: float
    commission: float
    pnl: Optional[float] = None           # realized pnl on SELL
    entry_price: Optional[float] = None   # filled entry price on SELL


@dataclass
class Position:
    symbol: str
    quantity: float
    average_entry_price: float
    entry_timestamp: datetime.datetime
    entry_index: int

    stop_price: float
    take_profit_price: float
    trailing_stop_price: float
    highest_price_since_entry: float


@dataclass
class ExecutionConfig:
    commission_rate: float = DEFAULT_COMMISSION_RATE
    slippage_bps: float = 2.0
    min_cash_buffer_pct: float = 0.01


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.01
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02
    trailing_stop_pct: float = 0.008
    max_position_pct_equity: float = 0.95
    time_stop_bars: int = 180
    cooldown_bars: int = 10
    max_drawdown_pct: float = 12.0


# ==========================================
# 3) Mock Data Generator
# ==========================================
def generate_mock_market_data(
    num_points: int = 500,
    base_price: float = 100.0,
    volatility: float = 0.002,
    trend: float = 0.0001,
    seed: Optional[int] = None,
    symbol: str = "BTC-USD",
) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(int(seed))
        random.seed(int(seed))

    dates = pd.date_range(start=datetime.datetime.now(), periods=num_points, freq="1min")

    rows = []
    current_price = float(base_price)
    for i in range(num_points):
        change_pct = np.random.normal(loc=trend, scale=volatility)
        current_price *= (1 + change_pct)

        open_p = current_price
        high_p = open_p * (1 + abs(np.random.normal(0, 0.001)))
        low_p = open_p * (1 - abs(np.random.normal(0, 0.001)))
        close_p = (open_p + high_p + low_p) / 3

        rows.append(
            dict(
                timestamp=dates[i],
                symbol=symbol,
                open=float(open_p),
                high=float(high_p),
                low=float(low_p),
                close=float(close_p),
                volume=int(np.random.randint(10, 100)),
            )
        )

    return pd.DataFrame(rows)


# ==========================================
# 4) Market Data Cursor
# ==========================================
class MarketDataCursor:
    def __init__(self, historical_data: pd.DataFrame):
        self._data = historical_data.sort_values("timestamp").reset_index(drop=True)
        self._current_index = 0
        self._max_index = len(self._data) - 1

    def has_next(self) -> bool:
        return self._current_index <= self._max_index

    def next_tick(self) -> Optional[pd.Series]:
        if not self.has_next():
            return None
        row = self._data.iloc[self._current_index]
        self._current_index += 1
        return row

    def get_history(self, window_size: int = 200) -> pd.DataFrame:
        end_idx = self._current_index
        start_idx = max(0, end_idx - window_size)
        return self._data.iloc[start_idx:end_idx]

    def reset(self):
        self._current_index = 0


# ==========================================
# 5) Strategies
# ==========================================
class BaseStrategy:
    name: str = "BaseStrategy"
    def generate_signal(self, history: pd.DataFrame) -> str:
        raise NotImplementedError


class MovingAverageCrossStrategy(BaseStrategy):
    def __init__(self, short_window: int, long_window: int):
        self.short_window = short_window
        self.long_window = long_window
        self.name = f"MA Cross ({short_window}/{long_window})"

    def generate_signal(self, history: pd.DataFrame) -> str:
        if len(history) < self.long_window + 2:
            return "HOLD"

        closes = history["close"].astype(float)
        short_ma = closes.rolling(window=self.short_window).mean()
        long_ma = closes.rolling(window=self.long_window).mean()

        prev_short, cur_short = short_ma.iloc[-2], short_ma.iloc[-1]
        prev_long, cur_long = long_ma.iloc[-2], long_ma.iloc[-1]

        if prev_short <= prev_long and cur_short > cur_long:
            return "BUY"
        if prev_short >= prev_long and cur_short < cur_long:
            return "SELL"
        return "HOLD"


class RSIStrategy(BaseStrategy):
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"RSI (Wilder {period}, {oversold}/{overbought})"

    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def generate_signal(self, history: pd.DataFrame) -> str:
        if len(history) < self.period + 3:
            return "HOLD"

        closes = history["close"].astype(float)
        rsi = self.calculate_rsi(closes, self.period)
        prev_rsi, cur_rsi = rsi.iloc[-2], rsi.iloc[-1]

        if prev_rsi <= self.oversold and cur_rsi > self.oversold:
            return "BUY"
        if prev_rsi >= self.overbought and cur_rsi < self.overbought:
            return "SELL"
        return "HOLD"


class MACDStrategy(BaseStrategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.name = f"MACD ({fast}/{slow}/{signal})"

    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = prices.ewm(span=self.fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=self.signal, adjust=False).mean()
        hist = macd - sig
        return macd, sig, hist

    def generate_signal(self, history: pd.DataFrame) -> str:
        if len(history) < (self.slow + self.signal + 3):
            return "HOLD"

        closes = history["close"].astype(float)
        macd, sig, _ = self.calculate_macd(closes)
        prev_macd, cur_macd = macd.iloc[-2], macd.iloc[-1]
        prev_sig, cur_sig = sig.iloc[-2], sig.iloc[-1]

        if prev_macd <= prev_sig and cur_macd > cur_sig:
            return "BUY"
        if prev_macd >= prev_sig and cur_macd < cur_sig:
            return "SELL"
        return "HOLD"


class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        self.name = f"Bollinger Bands ({period}, {std_dev}σ)"

    def calculate_bollinger(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = prices.rolling(window=self.period).mean()
        std = prices.rolling(window=self.period).std()
        upper = sma + std * self.std_dev
        lower = sma - std * self.std_dev
        return upper, sma, lower

    def generate_signal(self, history: pd.DataFrame) -> str:
        if len(history) < self.period + 3:
            return "HOLD"

        closes = history["close"].astype(float)
        upper, mid, lower = self.calculate_bollinger(closes)

        prev_price, cur_price = closes.iloc[-2], closes.iloc[-1]
        prev_lower, cur_lower = lower.iloc[-2], lower.iloc[-1]
        cur_upper = upper.iloc[-1]

        # bounce off lower band
        if prev_price <= prev_lower and cur_price > cur_lower:
            return "BUY"
        # touch upper band
        if cur_price >= cur_upper:
            return "SELL"
        return "HOLD"


# ==========================================
# 6) Regime Detection + Switching Strategy
# ==========================================
class Regime(str, Enum):
    TREND = "TREND"
    MEAN_REVERT = "MEAN_REVERT"
    HIGH_VOL = "HIGH_VOL"
    CHOP = "CHOP"


class RegimeDetector:
    """
    Regime signals:
    - HIGH_VOL: rolling std(returns) above threshold
    - TREND: abs(slope(log price)) above threshold AND efficiency ratio above threshold
    - MEAN_REVERT: default if not trend and not high vol
    """
    def __init__(
        self,
        lookback: int = 120,
        vol_lookback: int = 60,
        trend_slope_threshold: float = 0.00020,
        high_vol_threshold: float = 0.0040,
        er_threshold: float = 0.35,
    ):
        self.lookback = int(lookback)
        self.vol_lookback = int(vol_lookback)
        self.trend_slope_threshold = float(trend_slope_threshold)
        self.high_vol_threshold = float(high_vol_threshold)
        self.er_threshold = float(er_threshold)

    @staticmethod
    def _slope_of_log_price(prices: pd.Series) -> float:
        y = np.log(prices.values.astype(float))
        x = np.arange(len(y), dtype=float)
        xm = x.mean()
        ym = y.mean()
        denom = np.sum((x - xm) ** 2)
        if denom <= 0:
            return 0.0
        return float(np.sum((x - xm) * (y - ym)) / denom)

    @staticmethod
    def _efficiency_ratio(prices: pd.Series) -> float:
        diffs = prices.diff().dropna()
        if len(diffs) == 0:
            return 0.0
        net = abs(float(prices.iloc[-1]) - float(prices.iloc[0]))
        denom = float(diffs.abs().sum())
        return float(net / denom) if denom > 0 else 0.0

    def detect(self, history: pd.DataFrame) -> Regime:
        if history is None or len(history) < max(self.lookback, self.vol_lookback) + 3:
            return Regime.CHOP

        closes = history["close"].astype(float)
        rets = closes.pct_change().dropna()

        vol_window = rets.tail(self.vol_lookback) if len(rets) >= self.vol_lookback else rets
        vol = float(vol_window.std()) if len(vol_window) > 1 else 0.0

        if vol >= self.high_vol_threshold:
            return Regime.HIGH_VOL

        w = closes.tail(self.lookback)
        slope = abs(self._slope_of_log_price(w))
        er = self._efficiency_ratio(w)

        if slope >= self.trend_slope_threshold and er >= self.er_threshold:
            return Regime.TREND

        return Regime.MEAN_REVERT


class RegimeSwitchingStrategy(BaseStrategy):
    def __init__(
        self,
        detector: RegimeDetector,
        trend_strategy: BaseStrategy,
        mean_revert_strategy: BaseStrategy,
        high_vol_strategy: Optional[BaseStrategy] = None,
    ):
        self.detector = detector
        self.trend_strategy = trend_strategy
        self.mean_revert_strategy = mean_revert_strategy
        self.high_vol_strategy = high_vol_strategy
        self.last_regime: Regime = Regime.CHOP
        hv_name = high_vol_strategy.name if high_vol_strategy else "HOLD"
        self.name = f"RegimeSwitch({trend_strategy.name} | {mean_revert_strategy.name} | HV:{hv_name})"

    def generate_signal(self, history: pd.DataFrame) -> str:
        self.last_regime = self.detector.detect(history)

        if self.last_regime == Regime.TREND:
            return self.trend_strategy.generate_signal(history)

        if self.last_regime == Regime.MEAN_REVERT:
            return self.mean_revert_strategy.generate_signal(history)

        if self.last_regime == Regime.HIGH_VOL:
            return self.high_vol_strategy.generate_signal(history) if self.high_vol_strategy else "HOLD"

        return "HOLD"


# ==========================================
# 7) Advanced Portfolio Manager (risk sizing + stops + slippage + circuit breaker)
# ==========================================
class PortfolioManager:
    def __init__(
        self,
        initial_capital: float,
        execution: ExecutionConfig = ExecutionConfig(),
        risk: RiskConfig = RiskConfig(),
        enable_regime_risk: bool = True,
    ):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)

        self.positions: List[Position] = []
        self.trade_log: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime.datetime] = []

        self.execution = execution
        self.base_risk = risk
        self.enable_regime_risk = bool(enable_regime_risk)

        self._last_exit_index: Dict[str, int] = {}
        self._peak_equity: float = self.initial_capital
        self._halted: bool = False

        self._effective_risk: RiskConfig = risk  # refreshed each bar

    def _apply_regime_overrides(self, regime: Optional[Regime]) -> RiskConfig:
        """
        IMPORTANT: returns a fresh effective risk config every bar (no compounding).
        """
        r = self.base_risk
        if not self.enable_regime_risk or regime is None:
            return r

        # Tune multipliers here
        if regime == Regime.HIGH_VOL:
            return replace(
                r,
                risk_per_trade_pct=max(r.risk_per_trade_pct * 0.5, 0.001),
                stop_loss_pct=max(r.stop_loss_pct * 1.2, 0.003),
                trailing_stop_pct=max(r.trailing_stop_pct * 1.5, 0.002),
                take_profit_pct=max(r.take_profit_pct * 1.0, 0.005),
            )

        if regime == Regime.TREND:
            return replace(
                r,
                risk_per_trade_pct=max(r.risk_per_trade_pct * 1.0, 0.001),
                stop_loss_pct=max(r.stop_loss_pct * 1.0, 0.003),
                trailing_stop_pct=max(r.trailing_stop_pct * 1.1, 0.001),
            )

        if regime == Regime.MEAN_REVERT:
            return replace(
                r,
                risk_per_trade_pct=max(r.risk_per_trade_pct * 0.8, 0.001),
                stop_loss_pct=max(r.stop_loss_pct * 0.9, 0.002),
                take_profit_pct=max(r.take_profit_pct * 0.9, 0.004),
                trailing_stop_pct=max(r.trailing_stop_pct * 0.8, 0.001),
            )

        return r

    def set_regime(self, regime: Optional[Regime]):
        self._effective_risk = self._apply_regime_overrides(regime)

    def get_position(self, symbol: str) -> Optional[Position]:
        return next((p for p in self.positions if p.symbol == symbol), None)

    def _apply_slippage(self, side: str, mid_price: float) -> float:
        slip = (self.execution.slippage_bps / 10_000.0) * mid_price
        return mid_price + slip if side == "BUY" else mid_price - slip

    def _commission(self, notional: float) -> float:
        return notional * self.execution.commission_rate

    def _equity(self, current_price: float) -> float:
        pos_val = sum(p.quantity * current_price for p in self.positions)
        return self.cash + pos_val

    def update_equity(self, current_price: float, timestamp: datetime.datetime) -> float:
        eq = float(self._equity(current_price))
        self.equity_curve.append(eq)
        self.timestamps.append(timestamp)

        if eq > self._peak_equity:
            self._peak_equity = eq

        dd = (eq / self._peak_equity - 1) * 100
        if dd <= -abs(self._effective_risk.max_drawdown_pct):
            self._halted = True
        return eq

    def can_trade(self, symbol: str, bar_index: int) -> bool:
        if self._halted:
            return False
        last_exit = self._last_exit_index.get(symbol, -10**9)
        return (bar_index - last_exit) >= int(self._effective_risk.cooldown_bars)

    def _risk_position_size(self, price: float, stop_price: float, equity: float) -> float:
        risk_dollars = equity * float(self._effective_risk.risk_per_trade_pct)
        stop_dist = max(price - stop_price, 1e-9)
        qty = risk_dollars / stop_dist

        max_notional = equity * float(self._effective_risk.max_position_pct_equity)
        qty_cap = max_notional / price
        return max(0.0, min(qty, qty_cap))

    def _open_long(self, symbol: str, tick: pd.Series, bar_index: int):
        mid = float(tick["close"])
        ts = tick["timestamp"]

        entry_price = float(self._apply_slippage("BUY", mid))
        equity = float(self._equity(mid))

        stop_price = entry_price * (1 - float(self._effective_risk.stop_loss_pct))
        take_profit = entry_price * (1 + float(self._effective_risk.take_profit_pct))

        qty = float(self._risk_position_size(entry_price, stop_price, equity))
        if qty <= 0:
            return

        notional = qty * entry_price
        commission = self._commission(notional)
        buffer = float(self.execution.min_cash_buffer_pct) * equity

        if self.cash < (notional + commission + buffer):
            affordable = max(self.cash - commission - buffer, 0.0)
            qty = affordable / entry_price
            notional = qty * entry_price
            if qty <= 0:
                return
            commission = self._commission(notional)

        self.cash -= (notional + commission)

        pos = Position(
            symbol=symbol,
            quantity=qty,
            average_entry_price=entry_price,
            entry_timestamp=ts,
            entry_index=bar_index,
            stop_price=stop_price,
            take_profit_price=take_profit,
            trailing_stop_price=entry_price * (1 - float(self._effective_risk.trailing_stop_pct)),
            highest_price_since_entry=entry_price,
        )
        self.positions.append(pos)
        self.trade_log.append(Trade(ts, symbol, "BUY", entry_price, qty, commission))

    def _close_long(self, position: Position, tick: pd.Series, bar_index: int, reason: str = ""):
        mid = float(tick["close"])
        ts = tick["timestamp"]

        exit_price = float(self._apply_slippage("SELL", mid))
        qty = float(position.quantity)
        notional = qty * exit_price
        commission = self._commission(notional)

        self.cash += (notional - commission)
        pnl = (exit_price - float(position.average_entry_price)) * qty - commission

        self.positions.remove(position)
        self._last_exit_index[position.symbol] = int(bar_index)

        self.trade_log.append(
            Trade(
                timestamp=ts,
                symbol=position.symbol,
                side="SELL",
                price=exit_price,
                quantity=qty,
                commission=commission,
                pnl=float(pnl),
                entry_price=float(position.average_entry_price),
            )
        )

    def _update_trailing_stop(self, position: Position, current_price: float):
        if current_price > position.highest_price_since_entry:
            position.highest_price_since_entry = current_price
            new_trail = position.highest_price_since_entry * (1 - float(self._effective_risk.trailing_stop_pct))
            position.trailing_stop_price = max(position.trailing_stop_price, new_trail)

    def manage_open_positions(self, tick: pd.Series, bar_index: int):
        symbol = tick["symbol"]
        pos = self.get_position(symbol)
        if not pos:
            return

        price = float(tick["close"])
        self._update_trailing_stop(pos, price)

        # time stop
        if int(self._effective_risk.time_stop_bars) > 0 and (bar_index - pos.entry_index) >= int(self._effective_risk.time_stop_bars):
            self._close_long(pos, tick, bar_index, reason="TIME_STOP")
            return

        # exit priority: stop -> trailing -> take profit
        if price <= float(pos.stop_price):
            self._close_long(pos, tick, bar_index, reason="STOP_LOSS")
        elif price <= float(pos.trailing_stop_price):
            self._close_long(pos, tick, bar_index, reason="TRAIL_STOP")
        elif price >= float(pos.take_profit_price):
            self._close_long(pos, tick, bar_index, reason="TAKE_PROFIT")

    def execute_signal(self, signal: str, tick: pd.Series, bar_index: int):
        symbol = tick["symbol"]
        if not self.can_trade(symbol, bar_index):
            return

        pos = self.get_position(symbol)

        if signal == "BUY":
            if pos is None:
                self._open_long(symbol, tick, bar_index)

        elif signal == "SELL":
            if pos is not None:
                self._close_long(pos, tick, bar_index, reason="SIGNAL_SELL")


# ==========================================
# 8) Performance Analytics
# ==========================================
class PerformanceAnalyzer:
    @staticmethod
    def calculate_metrics(portfolio: PortfolioManager, initial_capital: float) -> Dict:
        if not portfolio.equity_curve:
            return {}

        eq = pd.Series(portfolio.equity_curve)
        rets = eq.pct_change().dropna()

        final_equity = float(portfolio.equity_curve[-1])
        total_return = ((final_equity - float(initial_capital)) / float(initial_capital)) * 100

        sell_trades = [t for t in portfolio.trade_log if t.side == "SELL" and t.pnl is not None]
        completed = [
            dict(
                entry=t.entry_price,
                exit=t.price,
                pnl=t.pnl,
                return_pct=((t.price - t.entry_price) / t.entry_price) * 100 if t.entry_price else 0.0,
            )
            for t in sell_trades
        ]

        win_rate = avg_win = avg_loss = profit_factor = 0.0
        if completed:
            wins = [t for t in completed if t["pnl"] > 0]
            losses = [t for t in completed if t["pnl"] < 0]
            win_rate = 100.0 * len(wins) / len(completed)
            avg_win = float(np.mean([t["pnl"] for t in wins])) if wins else 0.0
            avg_loss = float(np.mean([t["pnl"] for t in losses])) if losses else 0.0
            total_wins = float(sum(t["pnl"] for t in wins)) if wins else 0.0
            total_losses = float(abs(sum(t["pnl"] for t in losses))) if losses else 0.0
            profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        sharpe = 0.0
        max_dd = 0.0
        if len(rets) > 1 and rets.std() > 0:
            ann = np.sqrt(365 * 24 * 60)  # minute bars
            sharpe = float((rets.mean() / rets.std()) * ann)
            max_dd = float(((eq / eq.cummax()) - 1).min() * 100)

        return dict(
            total_return=float(total_return),
            final_equity=float(final_equity),
            total_trades=int(len(portfolio.trade_log)),
            completed_trades=int(len(completed)),
            win_rate=float(win_rate),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            profit_factor=float(profit_factor),
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_dd),
            equity_curve=eq,
            returns=rets,
        )


# ==========================================
# 9) Agent Orchestrator
# ==========================================
class TradingAgent:
    def __init__(self, strategy: BaseStrategy, portfolio: PortfolioManager, data_cursor: MarketDataCursor):
        self.strategy = strategy
        self.portfolio = portfolio
        self.cursor = data_cursor
        self.regimes: List[str] = []  # store regime per bar if available

    def run(self, progress_bar=None):
        bar_index = 0
        while self.cursor.has_next():
            tick = self.cursor.next_tick()
            if tick is None:
                break

            # Manage open positions first (stops, TP, trailing, time stop)
            self.portfolio.manage_open_positions(tick, bar_index)

            history = self.cursor.get_history(window_size=300)
            signal = self.strategy.generate_signal(history)

            # Regime-aware risk tuning (fresh each bar)
            regime = None
            if hasattr(self.strategy, "last_regime"):
                regime = getattr(self.strategy, "last_regime", None)
                self.regimes.append(regime.value if isinstance(regime, Regime) else str(regime))
            else:
                self.regimes.append("N/A")

            self.portfolio.set_regime(regime if isinstance(regime, Regime) else None)

            # Execute entry/exit signal
            self.portfolio.execute_signal(signal, tick, bar_index)

            # Mark equity and circuit breaker
            self.portfolio.update_equity(float(tick["close"]), tick["timestamp"])

            if progress_bar:
                denom = max(self.cursor._max_index, 1)
                progress_value = self.cursor._current_index / denom
                progress_bar.progress(min(progress_value, 1.0))

            bar_index += 1


# ==========================================
# 10) Streamlit App
# ==========================================
def build_strategy_from_ui() -> Tuple[BaseStrategy, Dict]:
    """
    Returns (strategy, extra_info_dict).
    extra_info_dict holds sub-strategies for plotting if needed.
    """
    use_regime = st.checkbox("Enable Regime Detection", value=True)

    # --- Build sub-strategies (trend + mean revert)
    st.subheader("Trend Strategy")
    trend_choice = st.selectbox("Trend Model", ["MA Cross", "MACD"], index=0)
    if trend_choice == "MA Cross":
        ma_short = st.slider("MA Short", 5, 60, 20)
        ma_long = st.slider("MA Long", 20, 250, 50)
        trend_strategy = MovingAverageCrossStrategy(ma_short, ma_long)
    else:
        macd_fast = st.slider("MACD Fast", 5, 25, 12)
        macd_slow = st.slider("MACD Slow", 15, 70, 26)
        macd_sig = st.slider("MACD Signal", 5, 25, 9)
        trend_strategy = MACDStrategy(macd_fast, macd_slow, macd_sig)

    st.subheader("Mean Reversion Strategy")
    mr_choice = st.selectbox("Mean Revert Model", ["Bollinger Bands", "RSI"], index=0)
    if mr_choice == "Bollinger Bands":
        bb_period = st.slider("BB Period", 10, 80, 20)
        bb_std = st.slider("BB Std Dev", 1.0, 4.0, 2.0)
        mr_strategy = BollingerBandsStrategy(bb_period, bb_std)
    else:
        rsi_period = st.slider("RSI Period", 5, 40, 14)
        oversold = st.slider("Oversold", 5, 45, 30)
        overbought = st.slider("Overbought", 55, 95, 70)
        mr_strategy = RSIStrategy(rsi_period, oversold, overbought)

    hv_behavior = st.selectbox("High Volatility Behavior", ["HOLD (stand down)", "Use Mean Revert Strategy"], index=0)
    hv_strategy = mr_strategy if hv_behavior.startswith("Use") else None

    # --- Regime detector
    detector = None
    if use_regime:
        st.subheader("Regime Detector")
        reg_lookback = st.slider("Regime Lookback (bars)", 60, 400, 120)
        reg_vol_lb = st.slider("Vol Lookback (bars)", 20, 200, 60)
        slope_th = st.slider("Trend Slope Threshold", 0.00005, 0.00100, 0.00020, 0.00005, format="%.5f")
        hv_th = st.slider("High Vol Threshold (std ret)", 0.0010, 0.0200, 0.0040, 0.0005, format="%.4f")
        er_th = st.slider("Efficiency Ratio Threshold", 0.10, 0.80, 0.35, 0.05)

        detector = RegimeDetector(
            lookback=int(reg_lookback),
            vol_lookback=int(reg_vol_lb),
            trend_slope_threshold=float(slope_th),
            high_vol_threshold=float(hv_th),
            er_threshold=float(er_th),
        )

        strategy = RegimeSwitchingStrategy(detector, trend_strategy, mr_strategy, hv_strategy)
    else:
        # If no regime detection, choose one main strategy quickly
        st.subheader("Single Strategy Mode")
        main_choice = st.selectbox("Main Strategy", ["Use Trend Strategy", "Use Mean Revert Strategy"], index=0)
        strategy = trend_strategy if main_choice.startswith("Use Trend") else mr_strategy

    return strategy, {
        "use_regime": use_regime,
        "trend_strategy": trend_strategy,
        "mr_strategy": mr_strategy,
        "detector": detector,
    }


def main():
    st.set_page_config(page_title="Titan Trading System", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    st.title("📈 Titan Trading System — Advanced Backtesting + Regime Detection")
    st.caption("Advanced execution (risk sizing + SL/TP/trailing/time stops + slippage) + regime-based strategy switching.")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Market Data")
        data_points = st.slider("Number of Data Points", 100, 8000, 1500)
        base_price = st.number_input("Base Price", 50.0, 500.0, 100.0)
        volatility = st.slider("Volatility", 0.0001, 0.02, 0.002, 0.0001)
        trend = st.slider("Trend", -0.003, 0.003, 0.0001, 0.0001)
        seed = st.number_input("Random Seed", value=42, step=1)

        st.markdown("---")
        st.header("🧠 Strategy")
        strategy, strat_info = build_strategy_from_ui()

        st.markdown("---")
        st.header("💼 Portfolio & Execution")

        initial_capital = st.number_input("Initial Capital ($)", 1000.0, 250000.0, 10000.0)

        st.subheader("Costs")
        commission_pct = st.slider("Commission (%)", 0.0, 1.0, 0.10) / 100
        slippage_bps = st.slider("Slippage (bps)", 0.0, 75.0, 2.0)
        buffer_pct = st.slider("Cash Buffer (%)", 0.0, 10.0, 1.0) / 100

        st.subheader("Risk")
        risk_pct = st.slider("Risk per Trade (% equity)", 0.1, 5.0, 1.0) / 100
        stop_loss_pct = st.slider("Stop Loss (%)", 0.1, 15.0, 1.0) / 100
        take_profit_pct = st.slider("Take Profit (%)", 0.1, 30.0, 2.0) / 100
        trail_pct = st.slider("Trailing Stop (%)", 0.1, 20.0, 0.8) / 100
        time_stop = st.slider("Time Stop (bars)", 0, 4000, 180)
        cooldown = st.slider("Cooldown (bars)", 0, 500, 10)
        max_dd = st.slider("Max Drawdown Halt (%)", 1.0, 60.0, 12.0)
        max_pos_pct = st.slider("Max Position (% equity)", 10, 100, 95) / 100

        enable_regime_risk = st.checkbox("Regime-adaptive risk sizing", value=True)

        st.markdown("---")
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            st.session_state.run_backtest = True

    # Run
    if st.session_state.get("run_backtest", False):
        with st.spinner("Running backtest..."):
            market_data = generate_mock_market_data(
                num_points=int(data_points),
                base_price=float(base_price),
                volatility=float(volatility),
                trend=float(trend),
                seed=int(seed),
                symbol="BTC-USD",
            )

            cursor = MarketDataCursor(market_data)

            execution = ExecutionConfig(
                commission_rate=float(commission_pct),
                slippage_bps=float(slippage_bps),
                min_cash_buffer_pct=float(buffer_pct),
            )
            risk = RiskConfig(
                risk_per_trade_pct=float(risk_pct),
                stop_loss_pct=float(stop_loss_pct),
                take_profit_pct=float(take_profit_pct),
                trailing_stop_pct=float(trail_pct),
                max_position_pct_equity=float(max_pos_pct),
                time_stop_bars=int(time_stop),
                cooldown_bars=int(cooldown),
                max_drawdown_pct=float(max_dd),
            )

            portfolio = PortfolioManager(
                initial_capital=float(initial_capital),
                execution=execution,
                risk=risk,
                enable_regime_risk=bool(enable_regime_risk),
            )

            progress = st.progress(0)
            agent = TradingAgent(strategy, portfolio, cursor)
            agent.run(progress)
            progress.empty()

            metrics = PerformanceAnalyzer.calculate_metrics(portfolio, float(initial_capital))

            # Attach regimes to market_data for plotting
            md = market_data.copy()
            md["regime"] = agent.regimes[: len(md)]

            st.session_state.backtest_results = dict(
                market_data=md,
                portfolio=portfolio,
                strategy=strategy,
                metrics=metrics,
                initial_capital=float(initial_capital),
                strat_info=strat_info,
            )
            st.session_state.run_backtest = False

    # Display
    if "backtest_results" not in st.session_state:
        st.info("👈 Configure in the sidebar, then click **Run Backtest**.")
        st.markdown(
            """
            ### What’s upgraded
            - **Regime Detection** (TREND / MEAN_REVERT / HIGH_VOL)
            - **Regime Switching Strategy** (trend strategy vs mean-revert strategy, optional stand-down)
            - **Advanced execution**: slippage + commission, risk-based sizing, SL/TP/trailing/time stop
            - **Circuit breaker**: halts trading after max drawdown breach
            """
        )
        return

    res = st.session_state.backtest_results
    market_data: pd.DataFrame = res["market_data"]
    portfolio: PortfolioManager = res["portfolio"]
    metrics: Dict = res["metrics"]
    strategy: BaseStrategy = res["strategy"]
    initial_capital = res["initial_capital"]

    st.subheader("📊 Performance Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%", delta=f"${metrics.get('final_equity', 0) - initial_capital:.2f}")
    with c2:
        st.metric("Final Equity", f"${metrics.get('final_equity', 0):.2f}")
    with c3:
        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%", delta=f"{metrics.get('completed_trades', 0)} exits")
    with c4:
        st.metric("Sharpe (1-min ann.)", f"{metrics.get('sharpe_ratio', 0):.2f}")
    with c5:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")

    st.markdown("---")

    # Equity + Returns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Equity Curve")
        if portfolio.equity_curve:
            equity_df = pd.DataFrame({"Timestamp": portfolio.timestamps, "Equity": portfolio.equity_curve})
            fig = px.line(equity_df, x="Timestamp", y="Equity", title=f"Equity Curve — {strategy.name}")
            fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray", annotation_text="Initial Capital")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("💰 Returns Distribution")
        rets = metrics.get("returns")
        if rets is not None and len(rets) > 0:
            fig = px.histogram(pd.DataFrame({"Returns": rets}), x="Returns", nbins=50, title="Distribution of Returns")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough returns to show distribution.")

    st.markdown("---")

    # Price chart with buys/sells
    st.subheader("📊 Price + Trades")
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Price", "Volume", "Regime"),
        row_heights=[0.60, 0.20, 0.20],
    )

    fig.add_trace(go.Scatter(x=market_data["timestamp"], y=market_data["close"], name="Close"), row=1, col=1)

    buy_trades = [t for t in portfolio.trade_log if t.side == "BUY"]
    sell_trades = [t for t in portfolio.trade_log if t.side == "SELL"]

    if buy_trades:
        fig.add_trace(
            go.Scatter(
                x=[t.timestamp for t in buy_trades],
                y=[t.price for t in buy_trades],
                mode="markers",
                name="BUY",
                marker=dict(symbol="triangle-up", size=10, color="green"),
            ),
            row=1, col=1
        )

    if sell_trades:
        fig.add_trace(
            go.Scatter(
                x=[t.timestamp for t in sell_trades],
                y=[t.price for t in sell_trades],
                mode="markers",
                name="SELL",
                marker=dict(symbol="triangle-down", size=10, color="red"),
            ),
            row=1, col=1
        )

    fig.add_trace(go.Bar(x=market_data["timestamp"], y=market_data["volume"], name="Volume"), row=2, col=1)

    # Regime timeline
    reg_order = ["TREND", "MEAN_REVERT", "HIGH_VOL", "CHOP", "N/A"]
    reg_map = {r: i for i, r in enumerate(reg_order)}
    reg_y = [reg_map.get(str(r), reg_map["N/A"]) for r in market_data.get("regime", ["N/A"] * len(market_data))]
    fig.add_trace(go.Scatter(x=market_data["timestamp"], y=reg_y, mode="lines", name="Regime"), row=3, col=1)
    fig.update_yaxes(
        row=3, col=1,
        tickmode="array",
        tickvals=list(reg_map.values()),
        ticktext=list(reg_map.keys()),
    )

    fig.update_layout(height=750, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Detailed metrics
    st.subheader("📋 Detailed Metrics")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Trade Statistics**")
        st.json(
            {
                "Total Trades": metrics.get("total_trades", 0),
                "Completed Trades (SELL exits)": metrics.get("completed_trades", 0),
                "Win Rate": f"{metrics.get('win_rate', 0):.2f}%",
                "Average Win": f"${metrics.get('avg_win', 0):.2f}",
                "Average Loss": f"${metrics.get('avg_loss', 0):.2f}",
                "Profit Factor": f"{metrics.get('profit_factor', 0):.2f}",
            }
        )

    with d2:
        st.markdown("**Risk & State**")
        st.json(
            {
                "Sharpe (1-min ann.)": f"{metrics.get('sharpe_ratio', 0):.2f}",
                "Max Drawdown": f"{metrics.get('max_drawdown', 0):.2f}%",
                "Cash Remaining": f"${portfolio.cash:.2f}",
                "Open Positions": len(portfolio.positions),
            }
        )

    st.subheader("📜 Trade History")
    if portfolio.trade_log:
        df = pd.DataFrame(
            [
                dict(
                    Timestamp=t.timestamp,
                    Side=t.side,
                    Symbol=t.symbol,
                    Price=f"${t.price:.2f}",
                    Quantity=f"{t.quantity:.6f}",
                    Commission=f"${t.commission:.2f}",
                    PnL=f"${t.pnl:.2f}" if t.pnl is not None else "—",
                )
                for t in portfolio.trade_log
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No trades executed.")


if __name__ == "__main__":
    main()
```
