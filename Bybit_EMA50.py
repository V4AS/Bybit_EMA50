import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta, time as dt_time
import time
import pandas_ta as ta
import vectorbt as vbt
import numpy as np
import plotly  # Ensure Plotly is installed: pip install plotly

# Fetch Historical Data from Bybit
def fetch_bybit_futures_data(symbol: str, interval: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """Fetch OHLCV data from Bybit futures API, capping end_time at now."""
    url = "https://api.bybit.com/v5/market/kline"
    end_time = min(end_time, datetime.now())
    data = []
    current_start = start_time
    max_days_per_request = 180

    while current_start < end_time:
        chunk_end = min(current_start + timedelta(days=max_days_per_request), end_time)
        start_ms = int(current_start.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000) - 1
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "start": start_ms,
            "end": end_ms,
            "limit": 1000
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        response_data = response.json()
        if response_data["retCode"] != 0:
            raise ValueError(f"API Error: {response_data['retMsg']}")
        candles = response_data["result"]["list"]
        if not candles:
            current_start = chunk_end
            continue
        data.extend(candles)
        current_start = chunk_end
        time.sleep(0.5)

    if not data:
        raise ValueError("No data fetched.")
    
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close"]].astype(float)
    return df.sort_index()

# Calculate Indicators
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA50, 20-period highs/lows, candle properties, and swing points."""
    df = df.copy()
    df["ema50"] = ta.ema(df["close"], length=50)
    df["high_20"] = df["high"].rolling(window=20, min_periods=1).max()
    df["low_20"] = df["low"].rolling(window=20, min_periods=1).min()
    df["is_green"] = df["close"] > df["open"]
    df["is_red"] = df["close"] < df["open"]
    df["body_size"] = df["close"] - df["open"]
    df["prev_swing_low"] = (df["low"].shift(1) < df["low"].shift(2)) & (df["low"].shift(1) < df["low"])
    df["prev_swing_high"] = (df["high"].shift(1) > df["high"].shift(2)) & (df["high"].shift(1) > df["high"])
    return df

# Identify Market Direction
def identify_market_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Detect bullish/bearish trends with a 5-candle lookback."""
    df = df.copy()
    df["bullish_trend"] = (df["close"] > df["ema50"]) & (df["high"].rolling(window=5).max() == df["high_20"])
    df["bearish_trend"] = (df["close"] < df["ema50"]) & (df["low"].rolling(window=5).min() == df["low_20"])
    return df

# Detect Entry Signals
def detect_entry_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Identify entries after trend and pullback using engulfing patterns."""
    df = df.copy()
    pattern1_long = df["is_green"] & (df["body_size"] > df["body_size"].shift(1).abs()) & (df["low"] < df["low"].shift(1))
    pattern2_long = df["is_green"] & (df["open"] < df["close"].shift(1)) & (df["close"] > df["open"].shift(1)) & df["prev_swing_low"]
    pattern1_short = df["is_red"] & (df["body_size"].abs() > df["body_size"].shift(1).abs()) & (df["high"] > df["high"].shift(1))
    pattern2_short = df["is_red"] & (df["open"] > df["close"].shift(1)) & (df["close"] < df["open"].shift(1)) & df["prev_swing_high"]
    df["long_entry"] = df["bullish_trend"].shift(3) & df["is_red"].shift(2) & df["is_red"].shift(1) & (pattern1_long | pattern2_long)
    df["short_entry"] = df["bearish_trend"].shift(3) & df["is_green"].shift(2) & df["is_green"].shift(1) & (pattern1_short | pattern2_short)
    return df

# Set Trade Management
def set_trade_management(df: pd.DataFrame) -> pd.DataFrame:
    """Define SL and TP with 1:2 risk-to-reward."""
    df = df.copy()
    df["sl_long"] = df["low"].where(df["long_entry"], np.nan)
    df["tp_long"] = (df["close"] + 2 * (df["close"] - df["sl_long"])).where(df["long_entry"], np.nan)
    df["sl_short"] = df["high"].where(df["short_entry"], np.nan)
    df["tp_short"] = (df["close"] - 2 * (df["sl_short"] - df["close"])).where(df["short_entry"], np.nan)
    return df

# Streamlit App
def main():
    st.title("Bybit EMA50 Cross Trading Strategy Backtest")
    st.write("Customize the symbol, timeframe, and date range, then click 'Run Backtest' to analyze the strategy.")

    # User Inputs
    st.subheader("Input Parameters")
    symbol = st.text_input("Futures Symbol (e.g., BTCUSDT):", value="BTCUSDT")
    interval = st.selectbox("Timeframe:", ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"], index=5)
    start_date = st.date_input("Start Date:", value=datetime(2023, 1, 1))
    end_date = st.date_input("End Date:", value=datetime.now())
    start_time = datetime.combine(start_date, dt_time.min)
    end_time = datetime.combine(end_date, dt_time.max)

    if st.button("Run Backtest"):
        try:
            with st.spinner('Fetching data from Bybit...'):
                df = fetch_bybit_futures_data(symbol, interval, start_time, end_time)
                df = calculate_indicators(df)
                df = identify_market_direction(df)
                df = detect_entry_signals(df)
                df = set_trade_management(df)
                df = df.dropna(subset=["ema50"])

            pf = vbt.Portfolio.from_signals(
                close=df["close"],
                entries=df["long_entry"].fillna(False),
                short_entries=df["short_entry"].fillna(False),
                sl_stop=df["sl_long"].combine_first(df["sl_short"]),
                tp_stop=df["tp_long"].combine_first(df["tp_short"]),
                price=df["close"],
                accumulate=False,
                freq=f"{interval}m" if interval.isdigit() else interval,
                init_cash=10000
            )

            # Performance Metrics
            st.subheader("Performance Metrics")
            st.write(f"**Total PnL:** ${pf.total_profit():.2f}")
            trades_df = pf.trades.records_readable
            winning_trades = trades_df[trades_df['PnL'] > 0]
            losing_trades = trades_df[trades_df['PnL'] < 0]
            avg_win = winning_trades['PnL'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['PnL'].mean() if not losing_trades.empty else 0
            risk_reward_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else np.inf
            st.write(f"**Risk to Reward Ratio:** {risk_reward_ratio:.2f}")
            st.write("### Detailed Statistics")
            st.dataframe(pf.stats())

            # Tables
            st.subheader("Trades, Positions, and Drawdowns")
            st.write("#### Trades")
            st.write("All executed trades with entry/exit details and PnL.")
            st.dataframe(trades_df)
            st.write("#### Positions")
            st.write("Records of opened and closed positions.")
            st.dataframe(pf.positions.records_readable)
            st.write("#### Drawdowns")
            st.write("Periods of portfolio decline from peak value.")
            st.dataframe(pf.drawdowns.records_readable)

            # Visualizations
            st.subheader("Visualizations")
            st.write("### Equity Curve")
            st.write("Portfolio equity over time.")
            st.plotly_chart(pf.plot())
            st.write("### Trade History")
            st.write("Entry and exit points for all trades.")
            st.plotly_chart(pf.trades.plot())
            st.write("### Position Sizes")
            st.write("Size of positions held over time.")
            st.plotly_chart(pf.positions.plot())
            st.write("### Order Executions")
            st.write("Timing and type of all orders.")
            st.plotly_chart(pf.orders.plot())
            st.write("### Drawdowns")
            st.write("Periods of equity decline.")
            st.plotly_chart(pf.drawdowns.plot())
            st.write("### Cumulative Cash Flow")
            st.write("Cumulative cash flow from trades.")
            st.plotly_chart(pf.plot_cash_flow())
            st.write("### Cash Balance")
            st.write("Available cash over time.")
            st.plotly_chart(pf.plot_cash())
            st.write("### Asset Value")
            st.write("Value of open positions over time.")
            st.plotly_chart(pf.plot_asset_value())
            st.write("### Underwater Plot")
            st.write("Percentage decline from peak equity.")
            st.plotly_chart(pf.plot_underwater())

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
