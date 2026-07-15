"""
OpenOptions - Professional Institutional Grade Quantitative Options Platform
Developed for Indian F&O (NSE) Market Analysis & EOD Backtesting.
"""

import datetime
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import scipy.stats as stat
import streamlit as strl
import yfinance as yf
from plotly.subplots import make_subplots

# ==============================================================================
# 1. ARCHITECTURE & LOGGING CONFIGURATION
# ==============================================================================
logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OpenOptions.Engine")

strl.set_page_config(
   page_title="OpenOptions | Institutional EOD Platform",
   page_icon="⚡",
   layout="wide",
   initial_sidebar_state="expanded",
)

# Professional Bloomberg/TradingView Dark Theme CSS
THEME_CSS = """
<style>
   @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');
   
   :root {
       --bg-main: #0B0E11;
       --bg-panel: #161A1E;
       --border-color: #23282E;
       --text-main: #E0E3EB;
       --text-muted: #848E9C;
       --accent: #F0B90B;
       --positive: #0ECB81;
       --negative: #F6465D;
       --info: #2196F3;
   }
   html, body, [data-testid="stAppViewContainer"] {
       background-color: var(--bg-main);
       color: var(--text-main);
       font-family: 'Inter', sans-serif;
   }
   [data-testid="stSidebar"] {
       background-color: #12161A !important;
       border-right: 1px solid var(--border-color);
   }
   .kpi-card {
       background-color: var(--bg-panel);
       border: 1px solid var(--border-color);
       border-radius: 8px;
       padding: 16px;
       box-shadow: 0 4px 12px rgba(0,0,0,0.5);
   }
   .kpi-title {
       font-size: 11px;
       text-transform: uppercase;
       letter-spacing: 1.2px;
       color: var(--text-muted);
       margin-bottom: 8px;
   }
   .kpi-value {
       font-family: 'JetBrains Mono', monospace;
       font-size: 22px;
       font-weight: 700;
       color: var(--text-main);
   }
   .kpi-positive { color: var(--positive); }
   .kpi-negative { color: var(--negative); }
   .kpi-accent { color: var(--accent); }
   
   .badge {
       padding: 4px 8px;
       border-radius: 4px;
       font-size: 11px;
       font-weight: 700;
       font-family: 'JetBrains Mono', monospace;
       text-transform: uppercase;
       display: inline-block;
   }
   .badge-elite { background: rgba(14,203,129,0.15); color: #0ECB81; border: 1px solid #0ECB81; }
   .badge-excellent { background: rgba(33,150,243,0.15); color: #2196F3; border: 1px solid #2196F3; }
   .badge-good { background: rgba(240,185,11,0.15); color: #F0B90B; border: 1px solid #F0B90B; }
   .badge-average { background: rgba(132,142,156,0.15); color: #848E9C; border: 1px solid #848E9C; }
   .badge-reject { background: rgba(246,70,93,0.15); color: #F6465D; border: 1px solid #F6465D; }
   
   h1, h2, h3, h4, h5 { font-family: 'Inter', sans-serif; font-weight: 600; }
   .stTabs [data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-weight: 500;}
</style>
"""
strl.markdown(THEME_CSS, unsafe_allow_html=True)

# ==============================================================================
# 2. CORE DATA STRUCTURES & DATACLASSES
# ==============================================================================
@dataclass
class OptionLeg:
   """Represents a single leg in an options strategy."""
   strike: float
   type_: str  # "CALL" or "PUT"
   side: str   # "BUY" or "SELL"
   dte: float
   price: float
   delta: float
   gamma: float
   theta: float
   vega: float
   iv: float

@dataclass
class StrategyMetrics:
   """Comprehensive metrics for an evaluated options strategy."""
   name: str
   net_credit: float
   net_debit: float
   max_profit: float
   max_loss: float
   breakevens: List[float]
   pop: float  # Probability of Profit
   risk_reward: float
   ev: float
   margin_req: float
   legs: List[OptionLeg]
   recommendation_score: float = 0.0
   annualized_roc: float = 0.0     # Return on capital employed/margin, annualized
   assignment_prob: float = 0.0    # For income strategies: probability the short leg finishes ITM
   regime_fit: str = ""            # How well this strategy's directional bias matches the detected regime

@dataclass
class MarketContext:
   """Contextual signals (trend, volatility regime, liquidity, expected move) used to grade
   whether a strategy actually fits current conditions - not just whether it prices well in
   isolation. This is what lets the engine treat a Bull Put Spread differently in a Bearish
   regime versus a Bullish one, instead of every strategy competing on raw EV alone."""
   direction: str        # "BULLISH", "BEARISH", "NEUTRAL"
   vol_regime: str        # "HIGH_IV", "LOW_IV", "NORMAL_IV"
   trend_strength: float  # 0-100, derived from ADX
   rs_score: float
   liquidity_score: float  # 0-100, derived from RVOL
   expected_move: float    # 1-sigma absolute price move over the option's DTE
   hv_percentile: float    # current HV_20 percentile rank vs its own trailing 1Y history

@dataclass
class TradeRecord:
   """Historical trade log entry for backtesting."""
   entry_date: pd.Timestamp
   exit_date: pd.Timestamp
   entry_price: float
   exit_price: float
   return_pct: float
   days_held: int

@dataclass
class BacktestStats:
   """Institutional backtest performance metrics."""
   total_trades: int
   win_rate: float
   cagr: float
   max_drawdown: float
   sharpe_ratio: float
   sortino_ratio: float
   profit_factor: float
   expectancy: float
   avg_win: float
   avg_loss: float
   equity_curve: pd.Series
   trade_log: List[TradeRecord]
   monthly_returns: pd.DataFrame

# ==============================================================================
# 3. EOD MARKET DATA LAYER
# ==============================================================================
class MarketDataGateway:
   """Handles EOD data extraction using caching and concurrency."""
   
   @staticmethod
   @strl.cache_data(ttl=86400)
   def fetch_universe() -> List[str]:
       """Returns liquid NSE F&O universe constituents."""
       return [
           "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "ITC", "LT", 
           "SBIN", "BHARTIARTL", "BAJFINANCE", "AXISBANK", "KOTAKBANK", "M&M", 
           "MARUTI", "SUNPHARMA", "ULTRACEMCO", "HINDUNILVR", "NTPC", "TATAMOTORS",
           "POWERGRID", "TATASTEEL", "COALINDIA", "BAJAJFINSV", "ONGC", "HCLTECH",
           "ASIANPAINT", "ADANIENT", "ADANIPORTS", "TITAN", "HAL", "JSWSTEEL",
           "INDUSINDBK", "GRASIM", "TECHM", "HINDALCO", "DIVISLAB", "CIPLA", 
           "DRREDDY", "BRITANNIA", "EICHERMOT", "APOLLOHOSP", "BAJAJ-AUTO",
           "SHRIRAMFIN", "WIPRO", "HEROMOTOCO", "BPCL", "TATACONSUM", "ZOMATO",
           "TRENT", "BEL", "PFC", "RECLTD", "VEDL", "GAIL", "AMBUJACEM", 
           "TVSMOTOR", "CHOLAFIN", "INDIGO", "PIDILITIND", "HDFCLIFE", "DLF", 
           "SBILIFE", "BANKBARODA", "PNB", "LTIM", "HINDCOPPER", "CUMMINSIND"
       ]

   @staticmethod
   @strl.cache_data(ttl=3600, show_spinner=False)
   def fetch_historical_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
       """Fetches and cleans OHLCV data for a given symbol."""
       try:
           ticker = "^NSEI" if symbol == "NIFTY50" else f"{symbol}.NS"
           df = yf.download(ticker, start=start_date, end=end_date, progress=False, group_by="ticker")
           if df.empty:
               return pd.DataFrame()
           
           # Handle yfinance multi-index structural changes. Rather than guessing the layout
           # from ticker membership alone (which silently produces the wrong columns if
           # yfinance changes its layout), detect which level actually holds OHLCV field
           # names and select that one.
           if isinstance(df.columns, pd.MultiIndex):
               field_names = {"open", "high", "low", "close", "adj close", "volume"}
               level0_vals = {str(x).lower() for x in df.columns.get_level_values(0)}
               level1_vals = {str(x).lower() for x in df.columns.get_level_values(1)}
               if field_names & level1_vals:
                   # (Ticker, Field) layout - fields live in level 1
                   if ticker in df.columns.get_level_values(0):
                       df = df.xs(ticker, axis=1, level=0)
                   else:
                       df.columns = df.columns.droplevel(0)
               elif field_names & level0_vals:
                   # (Field, Ticker) layout - fields live in level 0
                   if ticker in df.columns.get_level_values(1):
                       df = df.xs(ticker, axis=1, level=1)
                   else:
                       df.columns = df.columns.droplevel(1)
               else:
                   # Unrecognized layout - flatten to the innermost level defensively rather
                   # than guessing wrong; downstream req_cols check will catch a bad result.
                   df.columns = df.columns.get_level_values(-1)

           # Note: tz_localize(None) on an already tz-naive index is a documented pandas
           # no-op (verified), not a crash risk - this explicit check is for readability only.
           idx = pd.to_datetime(df.index)
           df.index = idx.tz_localize(None) if idx.tz is not None else idx
           df.columns = [col.capitalize() for col in df.columns]
           
           req_cols = ["Open", "High", "Low", "Close", "Volume"]
           if not all(col in df.columns for col in req_cols):
               return pd.DataFrame()
               
           return df[req_cols].dropna()
       except Exception as e:
           logger.error(f"Data fetch error for {symbol}: {e}")
           return pd.DataFrame()

# ==============================================================================
# 3B. LIVE NSE OPTION CHAIN GATEWAY
# ==============================================================================
class NSEOptionChainGateway:
   """Fetches the LIVE NSE option chain (strike-wise LTP, OI, OI change, volume, IV,
   bid/ask) directly from NSE's public JSON endpoints - the same data the NSE website
   itself renders. There is no official API key; the endpoints are unauthenticated but
   NSE actively rate-limits/blocks traffic that doesn't look like a real browser session.
   This is most likely to work reliably from a local/residential IP and may get blocked
   on cloud hosts (e.g. Streamlit Community Cloud). On ANY failure this returns an empty
   DataFrame - callers MUST fall back to OptionsPricingEngine's theoretical chain.
   """

   _INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "NIFTY50", "MIDCPNIFTY"}
   _HEADERS = {
       "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
       "Accept": "*/*",
       "Accept-Language": "en-US,en;q=0.9",
       "Accept-Encoding": "gzip, deflate, br",
       "Connection": "keep-alive",
   }

   @classmethod
   def _session(cls) -> requests.Session:
       s = requests.Session()
       s.headers.update(cls._HEADERS)
       # NSE requires a "warm-up" hit to the homepage to receive valid cookies before the
       # JSON API responds with real data instead of a 401/403.
       s.get("https://www.nseindia.com", timeout=5)
       return s

   @classmethod
   @strl.cache_data(ttl=180, show_spinner=False)
   def fetch_option_chain(cls, symbol: str) -> pd.DataFrame:
       """Returns the raw live chain (nearest expiry only) with columns:
       Strike, Type, Expiry, LTP, IV, OI, ChangeOI, Volume, BidPrice, AskPrice, Underlying.
       Empty DataFrame on any failure - caller must fall back to the theoretical chain."""
       sym = "NIFTY" if symbol == "NIFTY50" else symbol
       endpoint = "option-chain-indices" if sym in cls._INDEX_SYMBOLS else "option-chain-equities"
       url = f"https://www.nseindia.com/api/{endpoint}"
       try:
           session = cls._session()
           resp = session.get(url, params={"symbol": sym}, timeout=6)
           if resp.status_code != 200:
               logger.warning(f"NSE option chain fetch failed for {sym}: HTTP {resp.status_code}")
               return pd.DataFrame()
           data = resp.json()
           records = data.get("records", {})
           underlying = records.get("underlyingValue", 0.0)
           expiries = records.get("expiryDates", [])
           nearest_expiry = expiries[0] if expiries else None

           rows = []
           for item in records.get("data", []):
               if nearest_expiry and item.get("expiryDate") != nearest_expiry:
                   continue
               strike = item.get("strikePrice")
               for side, key in (("CALL", "CE"), ("PUT", "PE")):
                   leg = item.get(key)
                   if not leg:
                       continue
                   rows.append({
                       "Strike": strike, "Type": side, "Expiry": item.get("expiryDate"),
                       "LTP": leg.get("lastPrice", 0.0), "IV": leg.get("impliedVolatility", 0.0) / 100.0,
                       "OI": leg.get("openInterest", 0), "ChangeOI": leg.get("changeinOpenInterest", 0),
                       "Volume": leg.get("totalTradedVolume", 0),
                       "BidPrice": leg.get("bidprice", 0.0), "AskPrice": leg.get("askPrice", 0.0),
                       "Underlying": underlying,
                   })
           df = pd.DataFrame(rows)
           if not df.empty:
               df.attrs["underlying"] = underlying
               df.attrs["expiry"] = nearest_expiry
           return df
       except Exception as e:
           logger.error(f"NSE live chain error for {sym}: {e}")
           return pd.DataFrame()

   @staticmethod
   def to_strategy_chain(live_chain: pd.DataFrame, rfr: float = 0.065) -> pd.DataFrame:
       """Normalizes the live chain into the exact schema StrategyBuilder expects
       (Strike, Type, DTE, IV, price, delta, gamma, theta, vega, prob_itm), deriving
       Greeks from the live IV via BSM since NSE does not publish Greeks directly.
       Rows with zero/missing IV (illiquid strikes) are dropped rather than guessed."""
       if live_chain.empty:
           return pd.DataFrame()
       chain = live_chain[live_chain["IV"] > 0].copy()
       if chain.empty:
           return pd.DataFrame()

       S = float(chain["Underlying"].iloc[0])
       today = datetime.date.today()
       try:
           expiry_date = datetime.datetime.strptime(chain["Expiry"].iloc[0], "%d-%b-%Y").date()
           dte = max(1, (expiry_date - today).days)
       except (ValueError, TypeError):
           dte = 7

       # Prefer mid-of-bid-ask over LTP where a live two-sided market exists (LTP can be stale
       # on illiquid strikes); fall back to LTP when there's no valid quote.
       mid = (chain["BidPrice"] + chain["AskPrice"]) / 2.0
       chain["price"] = np.where((chain["BidPrice"] > 0) & (chain["AskPrice"] > 0), mid, chain["LTP"])
       chain["price"] = chain["price"].clip(lower=0.05)
       chain["DTE"] = dte

       T = dte / 365.25
       greeks = chain.apply(
           lambda r: OptionsPricingEngine.price_option(S, r["Strike"], T, rfr, r["IV"], r["Type"] == "CALL"),
           axis=1
       )
       for k in ("delta", "gamma", "theta", "vega", "prob_itm"):
           chain[k] = greeks.apply(lambda g: g[k])

       return chain[["Strike", "Type", "DTE", "IV", "price", "delta", "gamma", "theta", "vega",
                      "prob_itm", "OI", "ChangeOI", "Volume"]].reset_index(drop=True)

# ==============================================================================
# 3C. OPEN INTEREST ANALYTICS (PCR / MAX PAIN / OI WALLS)
# ==============================================================================
class OIAnalyticsEngine:
   """Open-Interest based analytics computed on the LIVE chain only - these are
   meaningless on the theoretical chain since it has no real OI data."""

   @staticmethod
   def compute_pcr(chain: pd.DataFrame) -> Dict[str, float]:
       calls, puts = chain[chain["Type"] == "CALL"], chain[chain["Type"] == "PUT"]
       call_oi, put_oi = float(calls["OI"].sum()), float(puts["OI"].sum())
       call_vol, put_vol = float(calls["Volume"].sum()), float(puts["Volume"].sum())
       return {
           "PCR_OI": (put_oi / call_oi) if call_oi > 0 else 0.0,
           "PCR_Volume": (put_vol / call_vol) if call_vol > 0 else 0.0,
           "Total_Call_OI": call_oi, "Total_Put_OI": put_oi,
       }

   @staticmethod
   def compute_max_pain(chain: pd.DataFrame) -> float:
       """The strike at which option writers' aggregate expiry payout is minimized -
       classic Max Pain theory. O(n^2) over strikes, trivial for a single expiry chain."""
       strikes = sorted(chain["Strike"].unique())
       calls = chain[chain["Type"] == "CALL"].set_index("Strike")["OI"]
       puts = chain[chain["Type"] == "PUT"].set_index("Strike")["OI"]
       best_strike, min_pain = strikes[0] if strikes else 0.0, float("inf")
       for E in strikes:
           call_pain = sum(max(0.0, E - K) * calls.get(K, 0) for K in strikes)
           put_pain = sum(max(0.0, K - E) * puts.get(K, 0) for K in strikes)
           total = call_pain + put_pain
           if total < min_pain:
               min_pain, best_strike = total, E
       return float(best_strike)

   @staticmethod
   def oi_walls(chain: pd.DataFrame, top_n: int = 3) -> Dict[str, List[Tuple[float, int]]]:
       """Strikes with the heaviest OI concentration - conventionally read as resistance
       (call OI) and support (put OI) zones, since that's where writers are most exposed."""
       calls = chain[chain["Type"] == "CALL"].nlargest(top_n, "OI")[["Strike", "OI"]]
       puts = chain[chain["Type"] == "PUT"].nlargest(top_n, "OI")[["Strike", "OI"]]
       return {
           "Resistance_Walls": list(zip(calls["Strike"].tolist(), calls["OI"].tolist())),
           "Support_Walls": list(zip(puts["Strike"].tolist(), puts["OI"].tolist())),
       }

# ==============================================================================
# 4. INSTITUTIONAL TECHNICAL ENGINE (VECTORIZED)
# ==============================================================================
class TechnicalEngine:
   """Calculates professional quantitative momentum and volatility factors."""

   @staticmethod
   def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
       """Wilder's smoothing: an EMA with alpha = 1/period. This is what ATR, RSI, and ADX
       are actually defined with; a plain simple moving average is a common but materially
       different approximation."""
       return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

   @classmethod
   def calc_atr(cls, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
       tr = pd.concat([
           high - low,
           (high - close.shift(1)).abs(),
           (low - close.shift(1)).abs()
       ], axis=1).max(axis=1)
       return cls._wilder_smooth(tr, period)

   @classmethod
   def calc_rsi(cls, series: pd.Series, period: int = 14) -> pd.Series:
       delta = series.diff()
       gain = cls._wilder_smooth(delta.where(delta > 0, 0.0), period)
       loss = cls._wilder_smooth(-delta.where(delta < 0, 0.0), period)
       rs = gain / (loss + 1e-9)
       return 100.0 - (100.0 / (1.0 + rs))

   @staticmethod
   def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
       ema_fast = close.ewm(span=fast, adjust=False).mean()
       ema_slow = close.ewm(span=slow, adjust=False).mean()
       macd = ema_fast - ema_slow
       signal_line = macd.ewm(span=signal, adjust=False).mean()
       return macd, signal_line, macd - signal_line

   @classmethod
   def calc_adx(cls, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
       up = high - high.shift(1)
       down = low.shift(1) - low
       plus_dm = np.where((up > down) & (up > 0), up, 0.0)
       minus_dm = np.where((down > up) & (down > 0), down, 0.0)
       
       atr = cls.calc_atr(high, low, close, period)
       plus_di = 100 * (cls._wilder_smooth(pd.Series(plus_dm, index=close.index), period) / atr)
       minus_di = 100 * (cls._wilder_smooth(pd.Series(minus_dm, index=close.index), period) / atr)
       
       dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
       return cls._wilder_smooth(dx, period)

   @staticmethod
   def calc_historical_volatility(close: pd.Series, period: int = 20) -> pd.Series:
       log_returns = np.log(close / close.shift(1))
       return log_returns.rolling(window=period).std() * np.sqrt(252)

   @staticmethod
   def calc_relvol_10d(volume: pd.Series, period: int = 10) -> pd.Series:
       """RelVol 10D%: today's volume vs its own 10-day average, as a % excess/deficit.
       (e.g. 30 = 30% above the 10-day average volume.) This column isn't part of the
       documented GreenStar script - this is the standard convention for '10D relative
       volume' used across most retail screeners."""
       avg_vol_10 = volume.rolling(period).mean()
       return (volume / (avg_vol_10 + 1e-9) - 1.0) * 100.0

   @staticmethod
   def calc_hod_vs_prior(high: pd.Series, atr: pd.Series) -> pd.Series:
       """%HoD vs Prior: today's high vs yesterday's high, ATR-normalized - same convention
       as the documented '%from LoD' column but applied to the high side. Also not part of
       the documented GreenStar script; this is the most natural reading of the column name
       given the sibling %from LoD formula it's paired with in the screenshot."""
       return ((high - high.shift(1)) / (atr + 1e-9)) * 100.0

   @staticmethod
   def calc_atr_pct(atr: pd.Series, close: pd.Series) -> pd.Series:
       """ATR 14d%: 14-day ATR expressed as a % of price - documented GreenStar column."""
       return (atr / (close + 1e-9)) * 100.0

   @staticmethod
   def calc_xfrom50(close: pd.Series, sma_50: pd.Series, atr: pd.Series) -> pd.Series:
       """xATR From 50: extension of price above/below the 50-day SMA, expressed in
       ATR-multiples - documented GreenStar column ('xFrom50')."""
       return (close - sma_50) / (atr + 1e-9)

   @staticmethod
   def calc_pct_from_lod(close: pd.Series, low: pd.Series, atr: pd.Series) -> pd.Series:
       """%ATR from LoD: distance of current price above today's session low, in
       ATR-normalized percentage terms - documented GreenStar column ('%from LoD')."""
       return ((close - low) / (atr + 1e-9)) * 100.0

   @staticmethod
   def calc_change_pct(close: pd.Series) -> pd.Series:
       """Change %: close vs prior session's close - documented GreenStar column ('Chg %')."""
       return (close / close.shift(1) - 1.0) * 100.0

   @staticmethod
   def calc_change_open_pct(close: pd.Series, open_: pd.Series) -> pd.Series:
       """Change Open %: current price vs today's own open - documented GreenStar column
       ('Chg Open %'). Guards against an exact-zero open (bad tick / illiquid penny stock),
       which otherwise produces silent +/-inf rather than a raised exception."""
       return (close / (open_ + 1e-9) - 1.0) * 100.0

   @staticmethod
   def calc_minervini_score(df: pd.DataFrame, rs_score: float) -> int:
       """Minervini Trend Template: the standard 8-point institutional checklist behind the
       'Minervini x/8' column. Needs 252 sessions of history (52-week hi/lo) and an RS
       Rating - this app uses the existing Mansfield-style RS score (0-100) as that input,
       consistent with how it's used everywhere else in the platform."""
       if len(df) < 252:
           return 0
       row = df.iloc[-1]
       c, sma50, sma150, sma200 = row['Close'], row['SMA_50'], row['SMA_150'], row['SMA_200']
       if pd.isna(sma50) or pd.isna(sma150) or pd.isna(sma200):
           return 0
       high_52w = df['High'].iloc[-252:].max()
       low_52w = df['Low'].iloc[-252:].min()
       sma200_prior = df['SMA_200'].iloc[-22] if len(df) > 22 and not pd.isna(df['SMA_200'].iloc[-22]) else sma200

       checks = [
           c > sma150 and c > sma200,                 # 1. Price above 150 & 200 SMA
           sma150 > sma200,                            # 2. 150 SMA above 200 SMA
           sma200 > sma200_prior,                      # 3. 200 SMA trending up (~1 month)
           sma50 > sma150 and sma50 > sma200,           # 4. 50 SMA above 150 & 200 SMA
           c > sma50,                                   # 5. Price above 50 SMA
           c >= low_52w * 1.25,                         # 6. At least 25% above 52-week low
           c >= high_52w * 0.75,                        # 7. Within 25% of 52-week high
           rs_score >= 70,                              # 8. RS Rating >= 70
       ]
       return int(sum(checks))

   @classmethod
   def apply_technicals(cls, df: pd.DataFrame) -> pd.DataFrame:
       """Applies all vectorized indicators to the OHLCV dataframe."""
       df = df.copy()
       c, h, l, v, o = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']
       
       df['SMA_50'] = c.rolling(50).mean()
       df['SMA_150'] = c.rolling(150).mean()
       df['SMA_200'] = c.rolling(200).mean()
       
       df['ATR'] = cls.calc_atr(h, l, c)
       df['RSI'] = cls.calc_rsi(c)
       df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = cls.calc_macd(c)
       df['ADX'] = cls.calc_adx(h, l, c)
       
       df['HV_20'] = cls.calc_historical_volatility(c, 20)
       df['HV_252'] = cls.calc_historical_volatility(c, 252)
       
       df['Vol_SMA_50'] = v.rolling(50).mean()
       df['RVOL'] = v / (df['Vol_SMA_50'] + 1e-9)
       df['RelVol_10D_Pct'] = cls.calc_relvol_10d(v)
       df['HoD_vs_Prior_Pct'] = cls.calc_hod_vs_prior(h, df['ATR'])
       df['ATR_Pct'] = cls.calc_atr_pct(df['ATR'], c)
       df['xFrom50'] = cls.calc_xfrom50(c, df['SMA_50'], df['ATR'])
       df['Pct_From_LoD'] = cls.calc_pct_from_lod(c, l, df['ATR'])
       df['Change_Pct'] = cls.calc_change_pct(c)
       df['Change_Open_Pct'] = cls.calc_change_open_pct(c, o)
       
       return df

class RelativeStrengthEngine:
   @staticmethod
   def calculate_rs(stock_df: pd.DataFrame, bench_df: pd.DataFrame) -> Tuple[float, float]:
       """Calculates Mansfield Relative Strength score and relative momentum."""
       if stock_df.empty or bench_df.empty: return 0.0, 0.0
       
       aligned = pd.concat([stock_df['Close'], bench_df['Close']], axis=1, join='inner')
       aligned.columns = ['Stock', 'Bench']
       if len(aligned) < 252: return 0.0, 0.0
       
       ratio = aligned['Stock'] / aligned['Bench']
       rs_line = (ratio / ratio.rolling(252).mean() - 1) * 100
       rs_momentum = rs_line.iloc[-1]
       
       perf_1y = (aligned['Stock'].iloc[-1] / aligned['Stock'].iloc[-252]) - 1
       bench_1y = (aligned['Bench'].iloc[-1] / aligned['Bench'].iloc[-252]) - 1
       outperformance = (perf_1y - bench_1y) * 100
       
       rs_score = max(0.0, min(100.0, 50.0 + (outperformance / 1.5)))
       return rs_score, rs_momentum

   @staticmethod
   def calculate_rs14(stock_df: pd.DataFrame, bench_df: pd.DataFrame) -> float:
       """RS14/SPY%: 14-session relative strength ratio vs the benchmark, expressed as a
       percentage (100 = performing exactly in line with the benchmark over 14 sessions;
       105 = outperforming by ~5 percentage points of ratio). 'SPY' in the column name is
       TradingView's US-market default benchmark; this app uses NIFTY50 as the equivalent
       Indian-market benchmark, consistent with the rest of the platform's RS calculations.
       Not part of the documented GreenStar script - standard relative-strength-ratio
       convention."""
       if stock_df.empty or bench_df.empty: return 100.0
       aligned = pd.concat([stock_df['Close'], bench_df['Close']], axis=1, join='inner')
       aligned.columns = ['Stock', 'Bench']
       if len(aligned) < 15: return 100.0
       
       stock_ret = aligned['Stock'].iloc[-1] / aligned['Stock'].iloc[-15]
       bench_ret = aligned['Bench'].iloc[-1] / aligned['Bench'].iloc[-15]
       if bench_ret == 0: return 100.0
       return float((stock_ret / bench_ret) * 100.0)

# ==============================================================================
# 5. WEINSTEIN STAGE ANALYSIS ENGINE
# ==============================================================================
class WeinsteinAnalyzer:
   """Professional implementation of Stan Weinstein's Market Phase Analysis."""
   
   @staticmethod
   def evaluate_stage(df_daily: pd.DataFrame) -> dict:
       """Calculates stage, slope, transition, and quality metrics using weekly aggregates."""
       df_weekly = df_daily.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
       if len(df_weekly) < 52:
           return {"Stage": 0, "Score": 0.0, "Confidence": 0.0, "Support": 0.0, "Resistance": 0.0}
           
       c = df_weekly['Close']
       ma30 = c.rolling(30).mean() # 30-week MA
       ma10 = c.rolling(10).mean() # 10-week MA
       
       # Calculate slope via linear regression over the last 10 weeks
       slopes = []
       x = np.arange(10)
       for i in range(len(ma30)):
           if i < 10 or pd.isna(ma30.iloc[i-10:i]).any():
               slopes.append(0.0)
           else:
               y = ma30.iloc[i-10:i].values
               slope, _ = np.polyfit(x, y, 1)
               normalized_slope = (slope / np.mean(y)) * 100 
               slopes.append(normalized_slope)
               
       df_weekly['Slope_30W'] = slopes
       
       curr_price = c.iloc[-1]
       curr_ma30 = ma30.iloc[-1]
       curr_ma10 = ma10.iloc[-1]
       slope_30w = slopes[-1]
       
       dist_from_30ma = ((curr_price - curr_ma30) / curr_ma30) * 100
       
       # Support/Resistance over recent 26 weeks (~6 months)
       recent_low = df_weekly['Low'].iloc[-26:].min()
       recent_high = df_weekly['High'].iloc[-26:].max()
       
       # Classification Engine
       stage = 4
       confidence = 0.0
       
       # Stage 2: Advancing (Price > MAs, MAs pointing up)
       if curr_price > curr_ma30 and curr_ma10 > curr_ma30 and slope_30w > 0.15:
           stage = 2
           confidence = min(100.0, 50.0 + (slope_30w * 25.0) - abs(dist_from_30ma - 10.0))
       # Stage 1: Basing (Price near MA, MA flat)
       elif abs(slope_30w) <= 0.15 and curr_price > curr_ma30 * 0.95:
           stage = 1
           confidence = 100 - abs(dist_from_30ma * 5)
       # Stage 4: Declining (Price < MA, MA down)
       elif curr_price < curr_ma30 and slope_30w < -0.15:
           stage = 4
           confidence = min(100.0, abs(slope_30w * 30))
       # Stage 3: Topping (Price near MA, MA flat but after uptrend)
       elif curr_price < curr_ma30 and abs(slope_30w) <= 0.15:
           stage = 3
           confidence = 100 - abs(dist_from_30ma * 5)
           
       return {
           "Stage": stage,
           "Quality_Score": max(0.0, min(100.0, confidence if stage == 2 else 0.0)),
           "Slope_30W": slope_30w,
           "Dist_MA": dist_from_30ma,
           "Support": recent_low,
           "Resistance": recent_high
       }

# ==============================================================================
# 5B. MARKET REGIME CLASSIFICATION
# ==============================================================================
class MarketRegimeEngine:
   """Classifies directional and volatility regime from signals the pipeline already
   computes (Stage, ADX, RS, HV percentile). Without this, every strategy type competes
   on raw EV/POP alone and a Bull Put Spread can outrank an Iron Condor on a stock that's
   actually in a Stage 4 downtrend - numerically fine, institutionally wrong."""

   @staticmethod
   def classify(stage: int, adx: float, rs_score: float, hv_percentile: float) -> Tuple[str, str]:
       if stage == 2 and adx >= 20 and rs_score >= 55:
           direction = "BULLISH"
       elif stage == 4 and adx >= 20 and rs_score <= 45:
           direction = "BEARISH"
       else:
           direction = "NEUTRAL"

       if hv_percentile >= 70:
           vol_regime = "HIGH_IV"
       elif hv_percentile <= 30:
           vol_regime = "LOW_IV"
       else:
           vol_regime = "NORMAL_IV"

       return direction, vol_regime

# ==============================================================================
# 6. QUANTITATIVE OPTIONS PRICING ENGINE (BSM)
# ==============================================================================
class OptionsPricingEngine:
   """Derives fair value option metrics using historical volatility and BSM model."""
   
   @staticmethod
   def d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
       if T <= 0 or sigma <= 0: return 0.0, 0.0
       d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
       d2 = d1 - sigma * np.sqrt(T)
       return d1, d2

   @classmethod
   def price_option(cls, S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> dict:
       """Returns Price, Delta, Gamma, Theta, Vega, and Prob ITM."""
       if T <= 0:
           price = max(0.0, S - K) if is_call else max(0.0, K - S)
           return {"price": price, "delta": 1.0 if (is_call and S>K) else (-1.0 if not is_call and S<K else 0.0), 
                   "gamma": 0.0, "theta": 0.0, "vega": 0.0, "prob_itm": 1.0 if price > 0 else 0.0}

       d1, d2 = cls.d1_d2(S, K, T, r, sigma)
       nd1, nd2 = stat.norm.cdf(d1), stat.norm.cdf(d2)
       n_d1, n_d2 = stat.norm.cdf(-d1), stat.norm.cdf(-d2)
       pdf_d1 = stat.norm.pdf(d1)

       gamma = pdf_d1 / (S * sigma * np.sqrt(T))
       vega = (S * np.sqrt(T) * pdf_d1) / 100.0
       
       if is_call:
           price = S * nd1 - K * np.exp(-r * T) * nd2
           delta = nd1
           theta = (-(S * sigma * pdf_d1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * nd2) / 365
           prob_itm = nd2
       else:
           price = K * np.exp(-r * T) * n_d2 - S * n_d1
           delta = nd1 - 1
           theta = (-(S * sigma * pdf_d1) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * n_d2) / 365
           prob_itm = n_d2

       return {
           "price": max(0.05, float(price)), "delta": float(delta), "gamma": float(gamma), 
           "theta": float(theta), "vega": float(vega), "prob_itm": float(prob_itm)
       }

   @classmethod
   def generate_theoretical_chain(cls, S: float, T_days: int, r: float, hv: float) -> pd.DataFrame:
       """Generates a matrix of theoretical strikes strictly derived from real Spot & real HV."""
       T = T_days / 365.25
       step = 5 if S < 500 else (10 if S < 1500 else (20 if S < 3000 else 50))
       atm = round(S / step) * step
       strikes = [atm + (i * step) for i in range(-25, 26)]
       
       options = []
       for K in strikes:
           if K <= 0: continue
           # Volatility smile heuristic: +10% IV per 10% OTM distance
           iv = hv * (1.0 + 0.1 * (abs(K - S) / S)) 
           
           call_metrics = cls.price_option(S, K, T, r, iv, True)
           put_metrics = cls.price_option(S, K, T, r, iv, False)
           
           options.append({"Strike": K, "Type": "CALL", "DTE": T_days, "IV": iv, **call_metrics})
           options.append({"Strike": K, "Type": "PUT", "DTE": T_days, "IV": iv, **put_metrics})
           
       return pd.DataFrame(options)

   @staticmethod
   def expected_move(S: float, sigma: float, T_days: int) -> float:
       """1-sigma expected absolute price move over T_days, from the same volatility
       used to price the chain. This is the reference every strike/width decision should
       be checked against - a spread that's tighter than the expected move is being sold
       for very little premium relative to how likely it is to be tested."""
       return S * sigma * np.sqrt(T_days / 365.25)

# ==============================================================================
# 7. MULTI-STRATEGY OPTIONS FACTORY
# ==============================================================================
class StrategyBuilder:
   """Evaluates and scores optimal standard option structures.

   Design notes vs. a naive single-delta-match engine:
   - Every spread strategy searches a range of short deltas AND a range of widths, scores
     every combination, and keeps the best - not the first strike that satisfies a delta filter.
   - Scoring blends POP, annualized return on margin, risk/reward, EV-on-margin, regime fit,
     and underlying liquidity - not just POP/RR/EV in isolation.
   - Margin for defined-risk credit spreads is approximated as max loss, which is how Indian
     broker SPAN+exposure margin behaves for a fully hedged spread; it is NOT a live broker
     margin figure and should be treated as an approximation.
   - A strategy is only returned if it clears a minimum quality floor (POP, RR, and blended
     score). If nothing clears the floor, evaluate_all returns an empty list rather than
     forcing a recommendation - the UI should treat that as "no qualifying trade today."
   """

   MIN_POP = 0.55
   MIN_RR = 0.12
   MIN_SCORE = 45.0
   WIDTH_STEPS = (1, 2, 3, 4)

   @staticmethod
   def _leg(row: pd.Series, side: str) -> OptionLeg:
       return OptionLeg(row["Strike"], row["Type"], side, row["DTE"], row["price"], row["delta"],
                         row["gamma"], row["theta"], row["vega"], row["IV"])

   @classmethod
   def _score_credit_spread(cls, credit: float, loss: float, pop: float, rr: float, margin: float,
                             dte: float, context: MarketContext, direction: str) -> Optional[float]:
       """Multi-factor score for a defined-risk credit spread. Returns None if the spread
       fails the minimum quality floor, so weak setups get discarded rather than ranked."""
       if pop < cls.MIN_POP or rr < cls.MIN_RR or margin <= 0:
           return None
       annualized_roc = (credit / margin) * (365.25 / dte) * 100
       ev = (credit * pop) - (loss * (1 - pop))
       ev_on_margin = max(0.0, (ev / margin) * 100)
       if direction == context.direction:
           regime_fit = 1.0
       elif context.direction == "NEUTRAL":
           regime_fit = 0.55
       else:
           regime_fit = 0.25  # fighting the detected trend
       liquidity_factor = context.liquidity_score / 100.0
       score = (pop * 100 * 0.30) + (min(annualized_roc, 120) / 120 * 100 * 0.20) \
               + (min(rr * 100, 50) * 0.15) + (min(ev_on_margin, 50) / 50 * 100 * 0.15) \
               + (regime_fit * 100 * 0.10) + (liquidity_factor * 100 * 0.10)
       return max(0.0, min(100.0, score))

   @classmethod
   def _score_income_strategy(cls, max_profit: float, margin: float, pop: float, dte: float,
                               context: MarketContext) -> Optional[float]:
       """Multi-factor score for stock+option income strategies (Covered Call / CSP)."""
       if pop < cls.MIN_POP or margin <= 0:
           return None
       annualized_roc = (max_profit / margin) * (365.25 / dte) * 100
       # Income-selling is a mildly-bullish-to-neutral posture; penalize selling into a
       # confirmed downtrend rather than treating every regime the same.
       regime_fit = 1.0 if context.direction in ("BULLISH", "NEUTRAL") else 0.35
       liquidity_factor = context.liquidity_score / 100.0
       score = (pop * 100 * 0.35) + (min(annualized_roc, 60) / 60 * 100 * 0.30) \
               + (regime_fit * 100 * 0.20) + (liquidity_factor * 100 * 0.15)
       return max(0.0, min(100.0, score))

   @classmethod
   def build_bull_put_spread(cls, chain: pd.DataFrame, S: float, context: MarketContext,
                              rfr: float = 0.065) -> Optional[StrategyMetrics]:
       puts = chain[chain["Type"] == "PUT"].sort_values("Strike").reset_index(drop=True)
       shorts = puts[(puts["delta"] > -0.40) & (puts["delta"] < -0.08) & (puts["Strike"] < S)]
       if shorts.empty: return None

       best, best_score = None, -1.0
       for _, short_data in shorts.iterrows():
           longs = puts[puts["Strike"] < short_data["Strike"]]
           if longs.empty: continue
           for n_back in cls.WIDTH_STEPS:
               if len(longs) < n_back: continue
               long_data = longs.iloc[-n_back]
               credit = short_data["price"] - long_data["price"]
               width = short_data["Strike"] - long_data["Strike"]
               if credit <= 0 or width <= 0: continue
               loss = width - credit
               if loss <= 0: continue
               pop = 1.0 - short_data["prob_itm"]
               rr = credit / loss
               margin = loss  # approximated as max loss for a defined-risk spread
               score = cls._score_credit_spread(credit, loss, pop, rr, margin, short_data["DTE"], context, "BULLISH")
               if score is None or score <= best_score: continue
               ev = (credit * pop) - (loss * (1 - pop))
               leg1, leg2 = cls._leg(short_data, "SELL"), cls._leg(long_data, "BUY")
               best_score = score
               best = StrategyMetrics("Bull Put Spread", credit, 0, credit, loss, [leg1.strike - credit],
                                       pop * 100, rr, ev, loss, [leg1, leg2], score,
                                       annualized_roc=(credit / margin) * (365.25 / short_data["DTE"]) * 100,
                                       regime_fit="BULLISH")
       return best

   @classmethod
   def build_bear_call_spread(cls, chain: pd.DataFrame, S: float, context: MarketContext,
                               rfr: float = 0.065) -> Optional[StrategyMetrics]:
       calls = chain[chain["Type"] == "CALL"].sort_values("Strike").reset_index(drop=True)
       shorts = calls[(calls["delta"] < 0.40) & (calls["delta"] > 0.08) & (calls["Strike"] > S)]
       if shorts.empty: return None

       best, best_score = None, -1.0
       for _, short_data in shorts.iterrows():
           longs = calls[calls["Strike"] > short_data["Strike"]]
           if longs.empty: continue
           for n_fwd in cls.WIDTH_STEPS:
               if len(longs) < n_fwd: continue
               long_data = longs.iloc[n_fwd - 1]
               credit = short_data["price"] - long_data["price"]
               width = long_data["Strike"] - short_data["Strike"]
               if credit <= 0 or width <= 0: continue
               loss = width - credit
               if loss <= 0: continue
               pop = 1.0 - short_data["prob_itm"]
               rr = credit / loss
               margin = loss
               score = cls._score_credit_spread(credit, loss, pop, rr, margin, short_data["DTE"], context, "BEARISH")
               if score is None or score <= best_score: continue
               ev = (credit * pop) - (loss * (1 - pop))
               leg1, leg2 = cls._leg(short_data, "SELL"), cls._leg(long_data, "BUY")
               best_score = score
               best = StrategyMetrics("Bear Call Spread", credit, 0, credit, loss, [leg1.strike + credit],
                                       pop * 100, rr, ev, loss, [leg1, leg2], score,
                                       annualized_roc=(credit / margin) * (365.25 / short_data["DTE"]) * 100,
                                       regime_fit="BEARISH")
       return best

   @classmethod
   def build_iron_condor(cls, chain: pd.DataFrame, S: float, context: MarketContext,
                          rfr: float = 0.065) -> Optional[StrategyMetrics]:
       puts = chain[chain["Type"] == "PUT"].sort_values("Strike").reset_index(drop=True)
       calls = chain[chain["Type"] == "CALL"].sort_values("Strike").reset_index(drop=True)
       put_shorts = puts[(puts["delta"] > -0.25) & (puts["delta"] < -0.10) & (puts["Strike"] < S)]
       call_shorts = calls[(calls["delta"] < 0.25) & (calls["delta"] > 0.10) & (calls["Strike"] > S)]
       if put_shorts.empty or call_shorts.empty: return None

       best, best_score = None, -1.0
       for _, ps in put_shorts.iterrows():
           put_longs = puts[puts["Strike"] < ps["Strike"]]
           if put_longs.empty: continue
           for _, cs in call_shorts.iterrows():
               call_longs = calls[calls["Strike"] > cs["Strike"]]
               if call_longs.empty: continue
               for n in cls.WIDTH_STEPS[:3]:
                   if len(put_longs) < n or len(call_longs) < n: continue
                   pl, cl = put_longs.iloc[-n], call_longs.iloc[n - 1]
                   put_width = ps["Strike"] - pl["Strike"]
                   call_width = cl["Strike"] - cs["Strike"]
                   credit = (ps["price"] - pl["price"]) + (cs["price"] - cl["price"])
                   width = max(put_width, call_width)
                   if credit <= 0 or width <= 0: continue
                   loss = width - credit
                   if loss <= 0: continue

                   be_lower, be_upper = ps["Strike"] - credit, cs["Strike"] + credit
                   T = ps["DTE"] / 365.25
                   # True interval POP: probability price finishes between both breakevens,
                   # not the product of two independently-computed leg POPs.
                   p_above_lower = OptionsPricingEngine.price_option(S, be_lower, T, rfr, ps["IV"], True)["prob_itm"]
                   p_above_upper = OptionsPricingEngine.price_option(S, be_upper, T, rfr, cs["IV"], True)["prob_itm"]
                   pop = p_above_lower - p_above_upper
                   if pop <= 0: continue

                   rr = credit / loss
                   margin = loss
                   score = cls._score_credit_spread(credit, loss, pop, rr, margin, ps["DTE"], context, "NEUTRAL")
                   if score is None: continue
                   # Penalize lopsided wings - a symmetric condor is the institutional default;
                   # a heavily skewed one is really a disguised directional bet.
                   symmetry_penalty = abs(put_width - call_width) / max(put_width, call_width)
                   score *= (1 - 0.15 * symmetry_penalty)
                   if score <= best_score: continue

                   ev = (credit * pop) - (loss * (1 - pop))
                   leg_ps, leg_pl = cls._leg(ps, "SELL"), cls._leg(pl, "BUY")
                   leg_cs, leg_cl = cls._leg(cs, "SELL"), cls._leg(cl, "BUY")
                   best_score = score
                   best = StrategyMetrics("Iron Condor", credit, 0, credit, loss, [be_lower, be_upper],
                                           pop * 100, rr, ev, loss, [leg_ps, leg_pl, leg_cs, leg_cl], score,
                                           annualized_roc=(credit / margin) * (365.25 / ps["DTE"]) * 100,
                                           regime_fit="NEUTRAL")
       return best

   @classmethod
   def build_covered_call(cls, chain: pd.DataFrame, S: float, context: MarketContext,
                           rfr: float = 0.065) -> Optional[StrategyMetrics]:
       calls = chain[chain["Type"] == "CALL"].sort_values("Strike")
       candidates = calls[(calls["delta"] < 0.45) & (calls["delta"] > 0.12) & (calls["Strike"] >= S * 0.98)]
       if candidates.empty: return None

       best, best_score = None, -1.0
       for _, short in candidates.iterrows():
           leg = cls._leg(short, "SELL")
           max_profit = (leg.strike - S) + leg.price
           max_loss = S - leg.price
           if max_profit <= 0 or max_loss <= 0: continue
           breakeven = max_loss
           T = leg.dte / 365.25
           pop = OptionsPricingEngine.price_option(S, breakeven, T, rfr, leg.iv, True)["prob_itm"]
           margin = S  # capital employed = cost of holding the underlying
           score = cls._score_income_strategy(max_profit, margin, pop, leg.dte, context)
           if score is None or score <= best_score: continue
           rr = max_profit / max_loss
           ev = (max_profit * pop) - (max_loss * (1 - pop))
           best_score = score
           best = StrategyMetrics("Covered Call", leg.price, 0, max_profit, max_loss, [breakeven],
                                   pop * 100, rr, ev, max_loss, [leg], score,
                                   annualized_roc=(max_profit / margin) * (365.25 / leg.dte) * 100,
                                   assignment_prob=short["prob_itm"] * 100, regime_fit="BULLISH/NEUTRAL")
       return best

   @classmethod
   def build_cash_secured_put(cls, chain: pd.DataFrame, S: float, context: MarketContext,
                               rfr: float = 0.065) -> Optional[StrategyMetrics]:
       puts = chain[chain["Type"] == "PUT"].sort_values("Strike")
       candidates = puts[(puts["delta"] > -0.45) & (puts["delta"] < -0.12) & (puts["Strike"] <= S * 1.02)]
       if candidates.empty: return None

       best, best_score = None, -1.0
       for _, short in candidates.iterrows():
           leg = cls._leg(short, "SELL")
           max_profit = leg.price
           max_loss = leg.strike - leg.price
           if max_loss <= 0: continue
           breakeven = max_loss
           margin = leg.strike  # capital secured = cash to buy the stock at strike if assigned
           pop = 1.0 - short["prob_itm"]
           score = cls._score_income_strategy(max_profit, margin, pop, leg.dte, context)
           if score is None or score <= best_score: continue
           rr = max_profit / max_loss
           ev = (max_profit * pop) - (max_loss * (1 - pop))
           best_score = score
           best = StrategyMetrics("Cash Secured Put", max_profit, 0, max_profit, max_loss, [breakeven],
                                   pop * 100, rr, ev, max_loss, [leg], score,
                                   annualized_roc=(max_profit / margin) * (365.25 / leg.dte) * 100,
                                   assignment_prob=short["prob_itm"] * 100, regime_fit="BULLISH/NEUTRAL")
       return best

   @classmethod
   def evaluate_all(cls, chain: pd.DataFrame, S: float, context: MarketContext,
                     rfr: float = 0.065) -> List[StrategyMetrics]:
       strats = [
           cls.build_bull_put_spread(chain, S, context, rfr),
           cls.build_bear_call_spread(chain, S, context, rfr),
           cls.build_iron_condor(chain, S, context, rfr),
           cls.build_covered_call(chain, S, context, rfr),
           cls.build_cash_secured_put(chain, S, context, rfr)
       ]
       # Overall quality floor: a strategy that individually cleared its own POP/RR bar can
       # still land below the blended-score floor once regime fit and liquidity are folded in.
       # Anything that doesn't clear it is dropped rather than force-ranked - "no trade" is a
       # valid outcome, not a failure of the engine.
       valid = [s for s in strats if s is not None and s.recommendation_score >= cls.MIN_SCORE]
       return sorted(valid, key=lambda x: x.recommendation_score, reverse=True)


# ==============================================================================
# 8. INSTITUTIONAL BACKTESTING ENGINE
# ==============================================================================
class WalkForwardBacktester:
   """Executes vectorized strategy backtesting with rigorous performance metrics."""
   
   @staticmethod
   def run_trend_backtest(df: pd.DataFrame, initial_capital: float = 100_000) -> BacktestStats:
       """Backtests a Mechanical Stage 2 Momentum Crossover Strategy."""
       if len(df) < 252:
           return BacktestStats(0,0,0,0,0,0,0,0,0,0, pd.Series(dtype=float), [], pd.DataFrame())
           
       c = df['Close']
       ma150 = c.rolling(150).mean()
       ma50 = c.rolling(50).mean()
       
       # Entry: 50 crosses above 150 AND price > 50
       entry_signals = (ma50 > ma150) & (ma50.shift(1) <= ma150.shift(1)) & (c > ma50)
       # Exit: Price closes below 150 MA
       exit_signals = (c < ma150) & (c.shift(1) >= ma150.shift(1))
       
       trades = []
       equity_vals = []
       current_cap = initial_capital  # realized capital as of the last closed trade
       in_trade = False
       entry_idx = 0
       entry_price = 0.0
       
       # Iterative log builder to strictly avoid look-ahead bias and capture dates
       for i in range(150, len(df)):
           date = df.index[i]
           price = c.iloc[i]
           
           if not in_trade and entry_signals.iloc[i]:
               in_trade = True
               entry_idx = i
               entry_price = price
           elif in_trade and exit_signals.iloc[i]:
               in_trade = False
               ret = (price - entry_price) / entry_price
               current_cap = current_cap * (1 + ret)
               days_held = (date - df.index[entry_idx]).days
               
               trades.append(TradeRecord(
                   entry_date=df.index[entry_idx],
                   exit_date=date,
                   entry_price=entry_price,
                   exit_price=price,
                   return_pct=ret,
                   days_held=days_held
               ))
           
           # Mark-to-market: while a position is open, reflect its unrealized P&L in the
           # equity curve rather than only updating capital at realized exits. Otherwise the
           # curve (and any Sharpe/Sortino/drawdown derived from it) is flat during every open
           # trade and only jumps on exit days, badly understating true volatility and risk.
           if in_trade:
               unrealized_ret = (price - entry_price) / entry_price
               equity_vals.append(current_cap * (1 + unrealized_ret))
           else:
               equity_vals.append(current_cap)
           
       # Prepend flat capital for the warm-up window (indices 0..149, before any signal is evaluated)
       padding = [initial_capital] * 150
       full_equity = padding + equity_vals
       equity_series = pd.Series(full_equity, index=df.index)
       
       if not trades:
           return BacktestStats(0,0,0,0,0,0,0,0,0,0, equity_series, [], pd.DataFrame())
           
       # Metrics Calculations
       returns = [t.return_pct for t in trades]
       wins = [r for r in returns if r > 0]
       losses = [r for r in returns if r <= 0]
       
       win_rate = (len(wins) / len(trades)) * 100
       avg_win = np.mean(wins) * 100 if wins else 0
       avg_loss = abs(np.mean(losses)) * 100 if losses else 0
       pf = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else float('inf')
       expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
       
       final_equity = equity_series.iloc[-1]
       years = (df.index[-1] - df.index[0]).days / 365.25
       cagr = ((final_equity / initial_capital) ** (1/years) - 1) * 100 if final_equity > 0 else -100.0
       
       running_max = equity_series.cummax()
       drawdown = (equity_series - running_max) / running_max
       max_dd = abs(drawdown.min()) * 100
       
       daily_returns = equity_series.pct_change().dropna()
       sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
       downside_std = daily_returns[daily_returns < 0].std()
       sortino = (daily_returns.mean() / downside_std) * np.sqrt(252) if downside_std != 0 else 0
       
       # Monthly Returns Matrix
       df_eq = pd.DataFrame({'Returns': daily_returns})
       monthly = df_eq.resample('ME').apply(lambda x: (x + 1).prod() - 1)
       monthly['Year'] = monthly.index.year
       monthly['Month'] = monthly.index.strftime('%b')
       monthly_pivot = monthly.pivot(index='Year', columns='Month', values='Returns').fillna(0) * 100
       month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
       monthly_pivot = monthly_pivot[[m for m in month_order if m in monthly_pivot.columns]]
       
       return BacktestStats(
           len(trades), win_rate, cagr, max_dd, sharpe, sortino, 
           pf, expectancy, avg_win, avg_loss, equity_series, trades, monthly_pivot
       )

# ==============================================================================
# 8B. SYNTHETIC OPTIONS STRATEGY BACKTESTER
# ==============================================================================
class OptionsStrategyBacktester:
   """Simulates a chosen options strategy, rolled systematically through history, mark-to-
   market daily using BSM with each day's declining time-to-expiry (IV per leg is frozen at
   entry - the same simplification every non-paid-data retail backtester makes).

   IMPORTANT: NSE does not publish historical option chains for free. This is a THEORETICAL
   reconstruction using historical spot prices + historical realized volatility as an IV
   proxy - it is NOT a replay of real historical option premiums. Treat the output as
   indicative of strategy *behavior* (how a Bull Put Spread tends to perform through trending
   vs. choppy history), not a promise of realizable historical returns.
   """

   _BUILDERS = {
       "Bull Put Spread": StrategyBuilder.build_bull_put_spread,
       "Bear Call Spread": StrategyBuilder.build_bear_call_spread,
       "Iron Condor": StrategyBuilder.build_iron_condor,
       "Covered Call": StrategyBuilder.build_covered_call,
       "Cash Secured Put": StrategyBuilder.build_cash_secured_put,
   }

   @staticmethod
   def _mtm_pnl(strat: StrategyMetrics, S0: float, S_now: float, T_now: float, rfr: float) -> float:
       """Per-unit unrealized P&L of the open position, marked to BSM fair value now."""
       pnl = 0.0
       for leg in strat.legs:
           if T_now > 0:
               px_now = OptionsPricingEngine.price_option(S_now, leg.strike, T_now, rfr, leg.iv, leg.type_ == "CALL")["price"]
           else:
               px_now = max(0.0, S_now - leg.strike) if leg.type_ == "CALL" else max(0.0, leg.strike - S_now)
           pnl += (leg.price - px_now) if leg.side == "SELL" else (px_now - leg.price)
       if strat.name == "Covered Call":
           pnl += (S_now - S0)  # the stock leg isn't in strat.legs, add it explicitly
       return pnl

   @classmethod
   def run(cls, df: pd.DataFrame, strategy_name: str, dte_target: int = 30,
           rfr: float = 0.065, initial_capital: float = 100_000.0) -> BacktestStats:
       builder_fn = cls._BUILDERS.get(strategy_name)
       empty = BacktestStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, pd.Series(dtype=float), [], pd.DataFrame())
       if builder_fn is None or "HV_20" not in df.columns or len(df) < 280:
           return empty

       trades: List[TradeRecord] = []
       equity_vals, equity_dates = [], []
       current_cap = initial_capital
       i, n = 252, len(df)

       while i < n - 5:
           S0 = float(df['Close'].iloc[i])
           hv0 = df['HV_20'].iloc[i]
           if pd.isna(hv0) or hv0 <= 0:
               i += 1
               continue

           chain0 = OptionsPricingEngine.generate_theoretical_chain(S0, dte_target, rfr, float(hv0))
           em0 = OptionsPricingEngine.expected_move(S0, float(hv0), dte_target)
           # Neutral, mid-range context for the backtest loop - the backtester is testing the
           # strategy's raw behavior, not the live regime-fit scoring used in the scanner.
           ctx = MarketContext("NEUTRAL", "NORMAL_IV", 50.0, 50.0, 100.0, em0, 50.0)
           strat = builder_fn(chain0, S0, ctx, rfr)
           if strat is None or strat.margin_req <= 0:
               i += 1
               continue

           exit_idx = min(i + dte_target, n - 1)
           cap_at_entry = current_cap
           last_pnl = 0.0
           for j in range(i, exit_idx + 1):
               S_j = float(df['Close'].iloc[j])
               days_left = max(0, dte_target - (j - i))
               T_j = days_left / 365.25
               last_pnl = cls._mtm_pnl(strat, S0, S_j, T_j, rfr)
               equity_vals.append(cap_at_entry + (last_pnl / strat.margin_req) * cap_at_entry)
               equity_dates.append(df.index[j])

           realized_ret = last_pnl / strat.margin_req
           current_cap = cap_at_entry * (1 + realized_ret)
           trades.append(TradeRecord(
               entry_date=df.index[i], exit_date=df.index[exit_idx],
               entry_price=S0, exit_price=float(df['Close'].iloc[exit_idx]),
               return_pct=realized_ret, days_held=(df.index[exit_idx] - df.index[i]).days
           ))
           i = exit_idx + 1  # non-overlapping rolls only

       if not trades:
           return empty

       equity_series = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_dates))
       equity_series = equity_series[~equity_series.index.duplicated(keep='last')]

       returns = pd.Series([t.return_pct for t in trades])
       win_rate = float((returns > 0).mean() * 100)
       daily_rets = equity_series.pct_change().dropna()
       sharpe = float((daily_rets.mean() / (daily_rets.std() + 1e-9)) * np.sqrt(252)) if len(daily_rets) > 5 else 0.0
       downside = daily_rets[daily_rets < 0]
       sortino = float((daily_rets.mean() / (downside.std() + 1e-9)) * np.sqrt(252)) if len(downside) > 5 else 0.0
       running_max = equity_series.cummax()
       drawdown = (equity_series - running_max) / running_max * 100
       max_dd = float(abs(drawdown.min())) if len(drawdown) else 0.0
       years = (equity_series.index[-1] - equity_series.index[0]).days / 365.25 if len(equity_series) > 1 else 1.0
       final_equity = equity_series.iloc[-1]
       cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 and final_equity > 0 else -100.0
       wins, losses = returns[returns > 0], returns[returns <= 0]
       profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float('inf')
       expectancy = float(returns.mean() * 100)
       avg_win = float(wins.mean() * 100) if len(wins) else 0.0
       avg_loss = float(losses.mean() * 100) if len(losses) else 0.0
       monthly = equity_series.resample('ME').last().pct_change().dropna() * 100

       return BacktestStats(len(trades), win_rate, cagr, max_dd, sharpe, sortino, profit_factor,
                             expectancy, avg_win, avg_loss, equity_series, trades, monthly.to_frame('Return'))

# ==============================================================================
# 8C. PORTFOLIO GREEKS & RISK MANAGEMENT
# ==============================================================================
@dataclass
class PortfolioPosition:
   """A single strategy sitting in the session's paper-trading book. NOT connected to a
   live broker or clearing account - this is an in-memory book that resets each session."""
   symbol: str
   strategy: StrategyMetrics
   lots: int
   lot_size: int
   added_on: str

class PortfolioRiskEngine:
   """Aggregates Greeks and capital-at-risk across every open position in the book."""

   @staticmethod
   def aggregate_greeks(positions: List[PortfolioPosition]) -> Dict[str, float]:
       net_delta = net_gamma = net_theta = net_vega = 0.0
       total_margin = total_max_loss = total_max_profit = 0.0
       for pos in positions:
           multiplier = pos.lots * pos.lot_size
           for leg in pos.strategy.legs:
               sign = 1 if leg.side == "BUY" else -1
               net_delta += sign * leg.delta * multiplier
               net_gamma += sign * leg.gamma * multiplier
               net_theta += sign * leg.theta * multiplier
               net_vega += sign * leg.vega * multiplier
           total_margin += pos.strategy.margin_req * multiplier
           total_max_loss += pos.strategy.max_loss * multiplier
           total_max_profit += pos.strategy.max_profit * multiplier
       return {
           "Net_Delta": net_delta, "Net_Gamma": net_gamma, "Net_Theta": net_theta,
           "Net_Vega": net_vega, "Total_Margin": total_margin,
           "Total_Max_Loss": total_max_loss, "Total_Max_Profit": total_max_profit,
       }

# ==============================================================================
# 9. MASTER SCANNER PIPELINE ORCHESTRATOR
# ==============================================================================
def process_symbol_pipeline(symbol: str, bench_df: pd.DataFrame, rfr: float = 0.065) -> Optional[dict]:
   """Master quantitative orchestrator merging Tech, Stage, and Options metrics."""
   try:
       end = datetime.datetime.now()
       start = end - datetime.timedelta(days=365*3) # 3 years for robust metrics
       
       df = MarketDataGateway.fetch_historical_data(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
       if df.empty or len(df) < 200: return None
       
       df = TechnicalEngine.apply_technicals(df)
       
       # Core Analytics
       stage_data = WeinsteinAnalyzer.evaluate_stage(df)
       rs_score, rs_mom = RelativeStrengthEngine.calculate_rs(df, bench_df)
       rs14_pct = RelativeStrengthEngine.calculate_rs14(df, bench_df)
       relvol_10d_pct = float(df['RelVol_10D_Pct'].iloc[-1]) if not pd.isna(df['RelVol_10D_Pct'].iloc[-1]) else 0.0
       hod_vs_prior_pct = float(df['HoD_vs_Prior_Pct'].iloc[-1]) if not pd.isna(df['HoD_vs_Prior_Pct'].iloc[-1]) else 0.0
       atr_pct = float(df['ATR_Pct'].iloc[-1]) if not pd.isna(df['ATR_Pct'].iloc[-1]) else 0.0
       x_from_50 = float(df['xFrom50'].iloc[-1]) if not pd.isna(df['xFrom50'].iloc[-1]) else 0.0
       pct_from_lod = float(df['Pct_From_LoD'].iloc[-1]) if not pd.isna(df['Pct_From_LoD'].iloc[-1]) else 0.0
       change_pct = float(df['Change_Pct'].iloc[-1]) if not pd.isna(df['Change_Pct'].iloc[-1]) else 0.0
       change_open_pct = float(df['Change_Open_Pct'].iloc[-1]) if not pd.isna(df['Change_Open_Pct'].iloc[-1]) else 0.0
       minervini_score = TechnicalEngine.calc_minervini_score(df, rs_score)
       
       curr_price = df['Close'].iloc[-1]
       hv = df['HV_20'].iloc[-1]
       atr = df['ATR'].iloc[-1]
       rvol = df['RVOL'].iloc[-1]
       adx = df['ADX'].iloc[-1]
       
       # A NaN or non-positive HV (halted stock, insufficient history, bad data) won't crash
       # the BSM engine, but it silently poisons every downstream Greek to NaN - and because
       # NaN comparisons are always False, the strategy quality floor then silently rejects
       # every candidate with no indication why. Skip the symbol explicitly instead.
       if pd.isna(hv) or hv <= 0:
           return None
       
       # Liquidity Filter (Min 5Cr Turnover)
       if df['Volume'].iloc[-1] * curr_price < 50_000_000:
           return None
           
       # Weighted Institutional Alpha Score Generation
       # Weights: Stage(25%), Trend/ADX(20%), RS(15%), Mom/RSI(15%), Liq/RVOL(15%), Vol(10%)
       stage_w = stage_data['Quality_Score'] * 0.25
       trend_w = min(100.0, max(0.0, adx * 2)) * 0.20
       rs_w = rs_score * 0.15
       mom_w = min(100.0, max(0.0, df['RSI'].iloc[-1] * 1.5)) * 0.15
       liq_w = min(100.0, rvol * 50) * 0.15
       vol_w = max(0.0, 100.0 - abs(hv - 0.25) * 100) * 0.10 # Prefers moderate HV (~25%)
       
       alpha_score = stage_w + trend_w + rs_w + mom_w + liq_w + vol_w
       
       if alpha_score >= 85: grade = "ELITE"
       elif alpha_score >= 70: grade = "EXCELLENT"
       elif alpha_score >= 55: grade = "GOOD"
       elif alpha_score >= 40: grade = "AVERAGE"
       else: grade = "REJECT"
       
       # Market Regime Classification (direction + volatility regime), so strategy selection
       # can be graded on fit rather than every strategy type competing on raw EV alone.
       hv_hist = df['HV_20'].dropna().iloc[-252:]
       hv_percentile = float((hv_hist < hv).mean() * 100) if len(hv_hist) >= 30 else 50.0
       direction, vol_regime = MarketRegimeEngine.classify(stage_data['Stage'], adx, rs_score, hv_percentile)
       liquidity_score = min(100.0, rvol * 50)
       expected_move = OptionsPricingEngine.expected_move(curr_price, hv, 30)
       context = MarketContext(
           direction=direction, vol_regime=vol_regime, trend_strength=min(100.0, adx * 2),
           rs_score=rs_score, liquidity_score=liquidity_score, expected_move=expected_move,
           hv_percentile=hv_percentile
       )
       
       # Options Matrix & Strategy Selection
       chain = OptionsPricingEngine.generate_theoretical_chain(curr_price, 30, rfr, hv)
       strategies = StrategyBuilder.evaluate_all(chain, curr_price, context, rfr)
       
       best_strat = strategies[0] if strategies else None
       
       return {
           "Symbol": symbol,
           "Spot": curr_price,
           "Stage": stage_data['Stage'],
           "Alpha_Score": alpha_score,
           "Grade": grade,
           "RS_Score": rs_score,
           "ADX": adx,
           "HV": hv * 100,
           "ATR": atr,
           "RVOL": rvol,
           "ATR_Pct": atr_pct,
           "xFrom50": x_from_50,
           "Pct_From_LoD": pct_from_lod,
           "Change_Pct": change_pct,
           "Change_Open_Pct": change_open_pct,
           "Minervini_Score": minervini_score,
           "RelVol_10D_Pct": relvol_10d_pct,
           "RS14_vs_Bench_Pct": rs14_pct,
           "HoD_vs_Prior_Pct": hod_vs_prior_pct,
           "Support": stage_data['Support'],
           "Resistance": stage_data['Resistance'],
           "Direction": direction,
           "Vol_Regime": vol_regime,
           "Expected_Move": expected_move,
           "Best_Strategy": best_strat.name if best_strat else "No Qualifying Trade",
           "Opt_Credit": best_strat.net_credit if best_strat else 0,
           "Opt_POP": best_strat.pop if best_strat else 0,
           "Opt_MaxLoss": best_strat.max_loss if best_strat else 0,
           "Opt_RR": best_strat.risk_reward if best_strat else 0,
           "Opt_EV": best_strat.ev if best_strat else 0,
           "Opt_Rec_Score": best_strat.recommendation_score if best_strat else 0,
           "Raw_DF": df,
           "Strategies": strategies
       }
   except Exception as e:
       logger.error(f"Error processing {symbol}: {e}")
       return None

# ==============================================================================
# 10. INSTITUTIONAL DASHBOARD UI RENDERING
# ==============================================================================
def render_payoff_chart(strategy: StrategyMetrics, spot: float) -> go.Figure:
   """Renders professional standard options payoff diagram."""
   lower = spot * 0.75
   upper = spot * 1.25
   x = np.linspace(lower, upper, 500)
   
   y = np.zeros_like(x)
   for leg in strategy.legs:
       multiplier = 1 if leg.side == "BUY" else -1
       if leg.type_ == "CALL":
           payoff = np.maximum(0, x - leg.strike) - leg.price
       else:
           payoff = np.maximum(0, leg.strike - x) - leg.price
       y += multiplier * payoff
       
   fig = go.Figure()
   fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', name='Payoff', line_color='#0ECB81' if y.mean() > 0 else '#F6465D'))
   fig.add_vline(x=spot, line_dash="dash", line_color="#848E9C", annotation_text="Current Spot")
   for be in strategy.breakevens:
       fig.add_vline(x=be, line_dash="dot", line_color="#F0B90B", annotation_text="BE")
       
   fig.update_layout(template="plotly_dark", title=f"Risk Profile: {strategy.name}", margin=dict(l=10,r=10,t=40,b=10), height=350)
   return fig

def main_app():
   # Application Header
   strl.markdown(
       """
       <div style="border-bottom:1px solid #23282E; padding-bottom:10px; margin-bottom:20px; display:flex; justify-content:space-between; align-items:center;">
           <div>
               <h1 style="margin:0; color:#E0E3EB;">OpenOptions Institutional</h1>
               <p style="margin:0; color:#848E9C; font-size:14px;">Quantitative EOD Scanning, Options Structuring & Walk-Forward Validation Platform</p>
           </div>
           <div><span class="badge badge-elite">PROD V2.0</span></div>
       </div>
       """, unsafe_allow_html=True
   )

   # Sidebar Controls
   with strl.sidebar:
       strl.markdown("### Matrix Configuration")
       scan_btn = strl.button("RUN EOD SCAN MATRIX", type="primary", use_container_width=True)
       
       strl.markdown("---")
       min_alpha = strl.slider("Minimum Alpha Score", 40, 95, 60)
       req_stage = strl.multiselect("Allowed Weinstein Stages", [1, 2, 3, 4], default=[2])
       rfr = strl.slider("India 10Y Yield (RFR %)", 5.0, 9.0, 6.75, 0.1) / 100.0
       
       strl.markdown("---")
       with strl.expander("🎯 Pine Screener Control Block (10 conditions)", expanded=False):
           strl.caption("Each condition can be toggled independently. 7 of these (ATR 14d%, xATR From 50, "
                        "%ATR from LoD, Change%, Change Open%, and Minervini x/8) match the documented "
                        "GreenStar script's formulas. RelVol 10D%, RS14/Bench%, and %HoD vs Prior are inferred "
                        "(not officially documented). RelVol At Time uses full-day RVOL as an EOD proxy - true "
                        "intraday time-of-day relative volume isn't computable from daily OHLCV data.")
           ps_filters = {}

           c1, c2 = strl.columns(2)
           ps_filters['atr_pct'] = (c1.checkbox("ATR 14d% >", key="ps_atr_on"),
                                     c2.number_input("", value=4.0, step=0.5, key="ps_atr_val", label_visibility="collapsed"))
           c1, c2 = strl.columns(2)
           ps_filters['xfrom50'] = (c1.checkbox("xATR From 50 <", key="ps_xf50_on"),
                                     c2.number_input("", value=5.0, step=0.5, key="ps_xf50_val", label_visibility="collapsed"))
           c1, c2, c3 = strl.columns([1, 1, 1])
           ps_lod_on = c1.checkbox("%ATR from LoD", key="ps_lod_on")
           ps_lod_lo = c2.number_input("Lo", value=0.0, step=5.0, key="ps_lod_lo", label_visibility="collapsed")
           ps_lod_hi = c3.number_input("Hi", value=65.0, step=5.0, key="ps_lod_hi", label_visibility="collapsed")
           ps_filters['pct_from_lod'] = (ps_lod_on, (ps_lod_lo, ps_lod_hi))
           c1, c2 = strl.columns(2)
           ps_filters['rvat'] = (c1.checkbox("RelVol At Time (EOD proxy) >", key="ps_rvat_on"),
                                  c2.number_input("", value=1.0, step=0.1, key="ps_rvat_val", label_visibility="collapsed"))
           c1, c2 = strl.columns(2)
           ps_filters['relvol10d'] = (c1.checkbox("RelVol 10D% >", key="ps_rv10_on"),
                                       c2.number_input("", value=30.0, step=5.0, key="ps_rv10_val", label_visibility="collapsed"))
           c1, c2 = strl.columns(2)
           ps_filters['rs14'] = (c1.checkbox("RS14/Bench% >", key="ps_rs14_on"),
                                  c2.number_input("", value=105.0, step=1.0, key="ps_rs14_val", label_visibility="collapsed"))
           c1, c2 = strl.columns(2)
           ps_filters['hod_vs_prior'] = (c1.checkbox("%HoD vs Prior >", key="ps_hod_on"),
                                          c2.number_input("", value=1.0, step=0.5, key="ps_hod_val", label_visibility="collapsed"))
           c1, c2 = strl.columns(2)
           ps_filters['change_pct'] = (c1.checkbox("Change% <", key="ps_chg_on"),
                                        c2.number_input("", value=7.0, step=0.5, key="ps_chg_val", label_visibility="collapsed"))
           c1, c2 = strl.columns(2)
           ps_filters['change_open_pct'] = (c1.checkbox("Change Open% >", key="ps_chgopen_on"),
                                             c2.number_input("", value=0.0, step=0.5, key="ps_chgopen_val", label_visibility="collapsed"))
           c1, c2 = strl.columns(2)
           ps_filters['minervini'] = (c1.checkbox("Minervini x/8 >", key="ps_mm_on"),
                                       c2.number_input("", value=5, step=1, key="ps_mm_val", label_visibility="collapsed"))
       
       strl.markdown("---")
       strl.info("💡 **Methodology**: Options pricing engine utilizes real EOD underlying prices and 20-day Realized Historical Volatility applied through the BSM model. No simulated placeholder data is used.")

   # State Init
   if "scan_results" not in strl.session_state:
       strl.session_state.scan_results = []
   
   # Global Benchmark Fetch
   end = datetime.datetime.now()
   start = end - datetime.timedelta(days=365*3)
   bench_df = MarketDataGateway.fetch_historical_data("NIFTY50", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
   if bench_df.empty:
       strl.sidebar.warning("⚠️ NIFTY50 benchmark data unavailable. Relative Strength scores will read 0 for this session.")

   # Scanner Execution
   if scan_btn:
       universe = MarketDataGateway.fetch_universe()
       results = []
       
       prog = strl.progress(0)
       stat_txt = strl.empty()
       
       with ThreadPoolExecutor(max_workers=10) as executor:
           futures = {executor.submit(process_symbol_pipeline, sym, bench_df, rfr): sym for sym in universe}
           for i, fut in enumerate(as_completed(futures)):
               sym = futures[fut]
               stat_txt.markdown(f"🔬 Compiling Quant Vectors: **{sym}**")
               prog.progress((i + 1) / len(universe))
               res = fut.result()
               if res: results.append(res)
               
       prog.empty()
       stat_txt.empty()
       strl.session_state.scan_results = results

   raw_results = strl.session_state.scan_results
   if not raw_results:
       strl.warning("System Idle. Initialize EOD Scan Matrix via sidebar to begin.")
       return

   df_res = pd.DataFrame(raw_results)
   # Apply user filters
   filtered_df = df_res[(df_res["Alpha_Score"] >= min_alpha) & (df_res["Stage"].isin(req_stage))]

   # Pine Screener Control Block: generic application of each toggled condition
   _ps_column_map = {
       'atr_pct': ('ATR_Pct', 'gt'), 'xfrom50': ('xFrom50', 'lt'),
       'pct_from_lod': ('Pct_From_LoD', 'between'), 'rvat': ('RVOL', 'gt'),
       'relvol10d': ('RelVol_10D_Pct', 'gt'), 'rs14': ('RS14_vs_Bench_Pct', 'gt'),
       'hod_vs_prior': ('HoD_vs_Prior_Pct', 'gt'), 'change_pct': ('Change_Pct', 'lt'),
       'change_open_pct': ('Change_Open_Pct', 'gt'), 'minervini': ('Minervini_Score', 'gt'),
   }
   if not filtered_df.empty:
       for key, (enabled, val) in ps_filters.items():
           if not enabled: continue
           col, mode = _ps_column_map[key]
           if mode == 'gt':
               filtered_df = filtered_df[filtered_df[col] > val]
           elif mode == 'lt':
               filtered_df = filtered_df[filtered_df[col] < val]
           elif mode == 'between':
               lo, hi = val
               filtered_df = filtered_df[(filtered_df[col] >= lo) & (filtered_df[col] <= hi)]

   # Tabs Generation
   t_dash, t_scan, t_deep, t_back, t_chain, t_port = strl.tabs([
       "📊 Market Dashboard", "⚡ Opportunity Scanner", "🔍 Deep Dive Strategy",
       "🕰️ Backtest Engine", "⛓️ Live Option Chain", "📦 Portfolio Greeks"
   ])

   # --- TAB 1: DASHBOARD ---
   with t_dash:
       c1, c2, c3, c4 = strl.columns(4)
       c1.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Universe Scanned</div><div class='kpi-value'>{len(df_res)}</div></div>", unsafe_allow_html=True)
       c2.markdown(f"<div class='kpi-card'><div class='kpi-title'>Qualified Targets</div><div class='kpi-value'>{len(filtered_df)}</div></div>", unsafe_allow_html=True)
       
       avg_score = df_res['Alpha_Score'].mean()
       c3.markdown(f"<div class='kpi-card'><div class='kpi-title'>Market Average Alpha</div><div class='kpi-value'>{avg_score:.1f}</div></div>", unsafe_allow_html=True)
       
       top_sym = filtered_df.sort_values('Alpha_Score', ascending=False).iloc[0]['Symbol'] if not filtered_df.empty else "N/A"
       c4.markdown(f"<div class='kpi-card'><div class='kpi-title'>Top Alpha Target</div><div class='kpi-value kpi-positive'>{top_sym}</div></div>", unsafe_allow_html=True)

       strl.markdown("<br>", unsafe_allow_html=True)
       col_c1, col_c2 = strl.columns(2)
       with col_c1:
           stage_counts = df_res['Stage'].value_counts().reset_index()
           stage_counts.columns = ['Stage', 'Count']
           fig_pie = px.pie(stage_counts, values='Count', names='Stage', title="Weinstein Market Breadth Distribution", hole=0.4, template="plotly_dark", color_discrete_sequence=['#F6465D', '#F0B90B', '#2196F3', '#0ECB81'])
           strl.plotly_chart(fig_pie, use_container_width=True)
       with col_c2:
           fig_scat = px.scatter(df_res, x='Alpha_Score', y='RS_Score', color='Grade', hover_name='Symbol', title="Alpha vs Relative Strength Scatter Matrix", template="plotly_dark")
           strl.plotly_chart(fig_scat, use_container_width=True)

   # --- TAB 2: SCANNER ---
   with t_scan:
       if filtered_df.empty:
           strl.warning("No assets met the strict filtering parameters.")
       else:
           display_cols = ['Symbol', 'Spot', 'Grade', 'Stage', 'Direction', 'Alpha_Score', 'RS_Score', 'ADX', 'HV',
                            'ATR_Pct', 'xFrom50', 'Pct_From_LoD', 'RVOL', 'RelVol_10D_Pct', 'RS14_vs_Bench_Pct',
                            'HoD_vs_Prior_Pct', 'Change_Pct', 'Change_Open_Pct', 'Minervini_Score',
                            'Best_Strategy', 'Opt_POP']
           d_df = filtered_df[display_cols].copy().sort_values('Alpha_Score', ascending=False)
           
           # Formatting for display
           for c in ['Spot', 'Alpha_Score', 'RS_Score', 'ADX', 'HV']:
               d_df[c] = d_df[c].round(2)
           d_df['Opt_POP'] = d_df['Opt_POP'].round(1).astype(str) + "%"
           
           strl.dataframe(d_df, use_container_width=True, hide_index=True)
           
           csv = d_df.to_csv(index=False).encode('utf-8')
           strl.download_button("Export Scan Matrix (CSV)", data=csv, file_name=f"openoptions_scan_{datetime.date.today()}.csv", mime="text/csv")

   # --- TAB 3: DEEP DIVE ---
   with t_deep:
       if not filtered_df.empty:
           tgt = strl.selectbox("Select Target Asset for Institutional Diagnosis:", filtered_df.sort_values('Alpha_Score', ascending=False)['Symbol'].tolist())
           row = filtered_df[filtered_df['Symbol'] == tgt].iloc[0]
           raw_df = row['Raw_DF']
           
           col_chart, col_stats = strl.columns([2, 1])
           with col_chart:
               fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
               fig.add_trace(go.Candlestick(x=raw_df.index, open=raw_df['Open'], high=raw_df['High'], low=raw_df['Low'], close=raw_df['Close'], name='Price'), row=1, col=1)
               fig.add_trace(go.Scatter(x=raw_df.index, y=raw_df['SMA_50'], line=dict(color='#2196F3', width=1.5), name='50 SMA'), row=1, col=1)
               fig.add_trace(go.Scatter(x=raw_df.index, y=raw_df['SMA_150'], line=dict(color='#F0B90B', width=2), name='150 SMA (30W)'), row=1, col=1)
               fig.add_trace(go.Bar(x=raw_df.index, y=raw_df['Volume'], marker_color='#848E9C', name='Volume'), row=2, col=1)
               fig.update_layout(template="plotly_dark", title=f"{tgt} Market Microstructure", xaxis_rangeslider_visible=False, height=550, margin=dict(l=10,r=10,t=40,b=10))
               strl.plotly_chart(fig, use_container_width=True)
               
           with col_stats:
               grade_badge_map = {
                   "ELITE": "badge-elite", "EXCELLENT": "badge-excellent",
                   "GOOD": "badge-good", "AVERAGE": "badge-average", "REJECT": "badge-reject",
               }
               regime_badge_map = {"BULLISH": "badge-elite", "BEARISH": "badge-reject", "NEUTRAL": "badge-average"}
               badge_cls = grade_badge_map.get(row['Grade'], "badge-average")
               regime_cls = regime_badge_map.get(row['Direction'], "badge-average")
               strl.markdown(f"### {tgt} Profile Overview")
               strl.markdown(f"- **Alpha Grade**: <span class='badge {badge_cls}'>{row['Grade']}</span>", unsafe_allow_html=True)
               strl.markdown(f"- **Market Regime**: <span class='badge {regime_cls}'>{row['Direction']}</span> · `{row['Vol_Regime']}`", unsafe_allow_html=True)
               strl.markdown(f"- **Weinstein Stage**: `{row['Stage']}`")
               strl.markdown(f"- **Relative Strength Score**: `{row['RS_Score']:.1f}`")
               strl.markdown(f"- **ADX Trend Intensity**: `{row['ADX']:.1f}`")
               strl.markdown(f"- **Historical Volatility**: `{row['HV']:.1f}%`")
               strl.markdown(f"- **30D Expected Move**: `±₹{row['Expected_Move']:.2f}`")
               strl.markdown(f"- **Implied Floor (Support)**: `₹{row['Support']:.2f}`")
               
               strl.markdown("---")
               strl.markdown("### Options Strategy Ranking")
               if not row['Strategies']:
                   strl.warning("No strategy cleared the minimum quality floor (POP/risk-reward/regime fit) for this asset today.")
               for strat in row['Strategies'][:3]:
                   strl.markdown(f"**{strat.name}** - Score: `{strat.recommendation_score:.1f}`")
                   strl.markdown(f"<span style='font-size:12px;color:#848E9C;'>POP: {strat.pop:.1f}% | Ann. ROC: {strat.annualized_roc:.1f}% | EV: ₹{strat.ev:.2f} | Max Loss: ₹{strat.max_loss:.2f}</span>", unsafe_allow_html=True)
                   strl.markdown("<br>", unsafe_allow_html=True)

           # Detail section for top strategy
           top_strat = row['Strategies'][0] if row['Strategies'] else None
           if top_strat:
               strl.markdown("---")
               strl.markdown(f"### Optimal Strategy Execution: {top_strat.name}")
               
               s_c1, s_c2 = strl.columns([1, 2])
               with s_c1:
                   strl.info(f"**Leg Details:**")
                   for leg in top_strat.legs:
                       strl.markdown(f"- **{leg.side}** {leg.strike} {leg.type_} @ ₹{leg.price:.2f} (Δ: {leg.delta:.2f})")
                   strl.markdown(f"\n- **Net Credit/Debit:** `₹{top_strat.net_credit:.2f}` / `₹{top_strat.net_debit:.2f}`")
                   strl.markdown(f"- **Max Profit:** `₹{top_strat.max_profit:.2f}`")
                   strl.markdown(f"- **Max Risk:** `₹{top_strat.max_loss:.2f}`")
                   strl.markdown(f"- **Margin (approx., ≈ max loss for spreads):** `₹{top_strat.margin_req:.2f}`")
                   strl.markdown(f"- **Annualized ROC:** `{top_strat.annualized_roc:.1f}%`")
                   if top_strat.assignment_prob > 0:
                       strl.markdown(f"- **Assignment Probability:** `{top_strat.assignment_prob:.1f}%`")
                   strl.markdown(f"- **Expected Value:** `₹{top_strat.ev:.2f}`")
               with s_c2:
                   fig_p = render_payoff_chart(top_strat, row['Spot'])
                   strl.plotly_chart(fig_p, use_container_width=True)

               strl.markdown("---")
               pc1, pc2, pc3 = strl.columns([1, 1, 2])
               lots = pc1.number_input("Lots", min_value=1, value=1, step=1, key="add_lots")
               lot_size = pc2.number_input("Lot Size", min_value=1, value=1, step=1, key="add_lot_size",
                                            help="Enter the current NSE F&O lot size for this symbol - lot sizes change periodically and aren't hardcoded here to avoid using stale values.")
               if pc3.button("➕ Add to Portfolio Book", key="add_to_book"):
                   if "portfolio" not in strl.session_state:
                       strl.session_state["portfolio"] = []
                   strl.session_state["portfolio"].append(PortfolioPosition(
                       symbol=tgt, strategy=top_strat, lots=int(lots), lot_size=int(lot_size),
                       added_on=datetime.date.today().isoformat()
                   ))
                   strl.success(f"Added {tgt} {top_strat.name} to Portfolio Book. See the 'Portfolio Greeks' tab.")
           else:
               strl.info("No qualifying options strategy for this asset under current filters - this is an intentional 'no trade' outcome, not a missing feature.")

   # --- TAB 4: BACKTESTER ---
   with t_back:
       bt_mode = strl.radio("Backtest Mode", ["Stock Trend (Mechanical Stage 2 Breakout)", "Options Strategy (Synthetic BSM)"],
                             horizontal=True, key="bt_mode")

       if not df_res.empty:
           bt_tgt = strl.selectbox("Select Asset for Backtest Engine:", df_res['Symbol'].tolist(), key="bt_tgt")
           bt_row = df_res[df_res['Symbol'] == bt_tgt].iloc[0]

           if bt_mode.startswith("Stock"):
               strl.markdown("Evaluates Mechanical Stage 2 Momentum Breakout historical performance strictly avoiding look-ahead bias.")
               with strl.spinner("Executing Walk-Forward Matrix..."):
                   stats = WalkForwardBacktester.run_trend_backtest(bt_row['Raw_DF'])
           else:
               strl.warning("Synthetic BSM reconstruction using historical spot price + historical realized volatility as an "
                             "IV proxy. This is **not** a replay of real historical option premiums - NSE doesn't publish "
                             "historical option chains for free. Treat results as indicative of strategy behavior across "
                             "market regimes, not a promise of realizable historical P&L.")
               strl.caption("Each cycle risks 100% of current capital on margin (no position sizing/diversification, same "
                            "convention as the Stock Trend backtest) - a run of max-loss trades on a credit-selling strategy "
                            "can legitimately compound capital toward zero. That's the tail risk of undiversified premium "
                            "selling showing up honestly, not a bug in the simulation.")
               bt_strategy_name = strl.selectbox("Strategy to Backtest:", list(OptionsStrategyBacktester._BUILDERS.keys()), key="bt_strategy_name")
               bt_dte = strl.slider("Target DTE per roll cycle", 15, 45, 30, key="bt_dte")
               with strl.spinner("Simulating Options Strategy Roll..."):
                   stats = OptionsStrategyBacktester.run(bt_row['Raw_DF'], bt_strategy_name, dte_target=bt_dte)

           b1, b2, b3, b4, b5 = strl.columns(5)
           b1.metric("Total Trades", stats.total_trades)
           b2.metric("Win Rate", f"{stats.win_rate:.1f}%")
           b3.metric("CAGR", f"{stats.cagr:.2f}%")
           b4.metric("Max Drawdown", f"-{stats.max_drawdown:.2f}%")
           b5.metric("Sharpe Ratio", f"{stats.sharpe_ratio:.2f}")
           
           eq_c, log_c = strl.columns([2, 1])
           with eq_c:
               fig_eq = go.Figure()
               fig_eq.add_trace(go.Scatter(x=stats.equity_curve.index, y=stats.equity_curve.values, fill='tozeroy', line_color='#2196F3', name='Equity'))
               fig_eq.update_layout(template="plotly_dark", title=f"Equity Curve Simulation: {bt_tgt}", height=350, margin=dict(l=10,r=10,t=40,b=10))
               strl.plotly_chart(fig_eq, use_container_width=True)
               
               strl.markdown("#### Monthly Returns Heatmap (%)")
               if not stats.monthly_returns.empty:
                   fig_hm = px.imshow(stats.monthly_returns, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn", template="plotly_dark")
                   fig_hm.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10))
                   strl.plotly_chart(fig_hm, use_container_width=True)
               
           with log_c:
               strl.markdown("#### Institutional Trade Log")
               if stats.trade_log:
                   log_df = pd.DataFrame([vars(t) for t in stats.trade_log])
                   log_df['entry_date'] = log_df['entry_date'].dt.strftime('%Y-%m-%d')
                   log_df['exit_date'] = log_df['exit_date'].dt.strftime('%Y-%m-%d')
                   log_df['entry_price'] = log_df['entry_price'].round(2)
                   log_df['exit_price'] = log_df['exit_price'].round(2)
                   log_df['return_pct'] = (log_df['return_pct'] * 100).round(2).astype(str) + "%"
                   
                   strl.dataframe(log_df[['entry_date', 'return_pct', 'days_held']], use_container_width=True, hide_index=True)
                   
                   csv_log = log_df.to_csv(index=False).encode('utf-8')
                   strl.download_button("Export Full Trade Log", data=csv_log, file_name=f"{bt_tgt}_backtest_log.csv", mime="text/csv")
               else:
                   strl.info("No trades executed for this asset within parameters.")
       else:
           strl.warning("Run the EOD Scan Matrix via the sidebar to populate the backtest universe.")

   # --- TAB 5: LIVE OPTION CHAIN / OI ANALYTICS ---
   with t_chain:
       strl.markdown("### Live NSE Option Chain & Open Interest Analytics")
       strl.caption("Pulled live from NSE's public option-chain endpoints (not a paid data feed). NSE actively "
                     "rate-limits non-browser traffic, so this can fail intermittently or on cloud hosts - a "
                     "failed fetch here does not affect the Scanner, which always uses the theoretical chain.")

       chain_symbols = ["NIFTY", "BANKNIFTY"] + (df_res['Symbol'].tolist() if not df_res.empty else MarketDataGateway.fetch_universe())
       chain_sym = strl.selectbox("Symbol:", sorted(set(chain_symbols)), key="live_chain_symbol")

       if strl.button("🔄 Fetch Live Chain", key="fetch_live_chain"):
           with strl.spinner(f"Fetching live NSE option chain for {chain_sym}..."):
               live_chain = NSEOptionChainGateway.fetch_option_chain(chain_sym)
           strl.session_state["live_chain_df"] = live_chain
           strl.session_state["live_chain_symbol_fetched"] = chain_sym

       live_chain = strl.session_state.get("live_chain_df", pd.DataFrame())
       if live_chain.empty:
           strl.info("No live chain loaded yet - click 'Fetch Live Chain' above. If it keeps failing, NSE is "
                     "likely blocking this host's IP; the Scanner and Deep Dive tabs are unaffected since they "
                     "use the theoretical BSM chain instead.")
       else:
           underlying = live_chain.attrs.get("underlying", live_chain["Underlying"].iloc[0] if "Underlying" in live_chain else 0.0)
           expiry = live_chain.attrs.get("expiry", "N/A")
           pcr = OIAnalyticsEngine.compute_pcr(live_chain)
           max_pain = OIAnalyticsEngine.compute_max_pain(live_chain)
           walls = OIAnalyticsEngine.oi_walls(live_chain)

           oc1, oc2, oc3, oc4 = strl.columns(4)
           oc1.markdown(f"<div class='kpi-card'><div class='kpi-title'>Underlying Spot</div><div class='kpi-value'>₹{underlying:,.2f}</div></div>", unsafe_allow_html=True)
           oc2.markdown(f"<div class='kpi-card'><div class='kpi-title'>Nearest Expiry</div><div class='kpi-value'>{expiry}</div></div>", unsafe_allow_html=True)
           oc3.markdown(f"<div class='kpi-card'><div class='kpi-title'>PCR (OI)</div><div class='kpi-value'>{pcr['PCR_OI']:.2f}</div></div>", unsafe_allow_html=True)
           oc4.markdown(f"<div class='kpi-card'><div class='kpi-title'>Max Pain</div><div class='kpi-value'>₹{max_pain:,.0f}</div></div>", unsafe_allow_html=True)

           strl.markdown("<br>", unsafe_allow_html=True)
           wc1, wc2 = strl.columns(2)
           with wc1:
               strl.markdown("**Resistance Walls (heaviest Call OI)**")
               for strike, oi in walls["Resistance_Walls"]:
                   strl.markdown(f"- ₹{strike:,.0f} — OI: {oi:,.0f}")
           with wc2:
               strl.markdown("**Support Walls (heaviest Put OI)**")
               for strike, oi in walls["Support_Walls"]:
                   strl.markdown(f"- ₹{strike:,.0f} — OI: {oi:,.0f}")

           strl.markdown("---")
           strl.markdown("#### OI Distribution by Strike")
           oi_pivot = live_chain.pivot_table(index="Strike", columns="Type", values="OI", aggfunc="sum").fillna(0).sort_index()
           fig_oi = go.Figure()
           if "CALL" in oi_pivot.columns:
               fig_oi.add_trace(go.Bar(x=oi_pivot.index, y=oi_pivot["CALL"], name="Call OI", marker_color="#F6465D"))
           if "PUT" in oi_pivot.columns:
               fig_oi.add_trace(go.Bar(x=oi_pivot.index, y=oi_pivot["PUT"], name="Put OI", marker_color="#0ECB81"))
           fig_oi.add_vline(x=max_pain, line_dash="dash", line_color="#F0B90B", annotation_text="Max Pain")
           fig_oi.update_layout(template="plotly_dark", barmode="group", height=400, margin=dict(l=10, r=10, t=30, b=10))
           strl.plotly_chart(fig_oi, use_container_width=True)

           strl.markdown("#### Raw Chain")
           display_chain = live_chain[["Strike", "Type", "LTP", "IV", "OI", "ChangeOI", "Volume", "BidPrice", "AskPrice"]].sort_values(["Strike", "Type"])
           strl.dataframe(display_chain, use_container_width=True, hide_index=True)

           strat_chain = NSEOptionChainGateway.to_strategy_chain(live_chain)
           if not strat_chain.empty:
               strl.markdown("---")
               strl.markdown("#### Strategies Built On the LIVE Chain (real IV/OI, BSM-derived Greeks)")
               live_ctx = MarketContext("NEUTRAL", "NORMAL_IV", 50.0, 50.0, 100.0,
                                         OptionsPricingEngine.expected_move(underlying, strat_chain["IV"].median(), int(strat_chain["DTE"].iloc[0])),
                                         50.0)
               live_strats = StrategyBuilder.evaluate_all(strat_chain, underlying, live_ctx)
               if not live_strats:
                   strl.info("No strategy cleared the quality floor on the live chain right now.")
               for s in live_strats[:3]:
                   strl.markdown(f"**{s.name}** — Score: `{s.recommendation_score:.1f}` · POP: {s.pop:.1f}% · Ann. ROC: {s.annualized_roc:.1f}% · Max Loss: ₹{s.max_loss:.2f}")

   # --- TAB 6: PORTFOLIO GREEKS & RISK ---
   with t_port:
       strl.markdown("### Portfolio Greeks & Risk Management")
       strl.caption("A session-only paper book. Add strategies from the Deep Dive tab's 'Add to Portfolio' button. "
                     "This resets when the app session ends and is not connected to any broker or clearing account.")

       book: List[PortfolioPosition] = strl.session_state.get("portfolio", [])
       if not book:
           strl.info("Portfolio Book is empty. Go to Deep Dive Strategy → build a strategy → 'Add to Portfolio Book'.")
       else:
           agg = PortfolioRiskEngine.aggregate_greeks(book)
           g1, g2, g3, g4 = strl.columns(4)
           g1.markdown(f"<div class='kpi-card'><div class='kpi-title'>Net Delta</div><div class='kpi-value'>{agg['Net_Delta']:,.1f}</div></div>", unsafe_allow_html=True)
           g2.markdown(f"<div class='kpi-card'><div class='kpi-title'>Net Gamma</div><div class='kpi-value'>{agg['Net_Gamma']:,.3f}</div></div>", unsafe_allow_html=True)
           g3.markdown(f"<div class='kpi-card'><div class='kpi-title'>Net Theta (₹/day)</div><div class='kpi-value'>{agg['Net_Theta']:,.1f}</div></div>", unsafe_allow_html=True)
           g4.markdown(f"<div class='kpi-card'><div class='kpi-title'>Net Vega</div><div class='kpi-value'>{agg['Net_Vega']:,.1f}</div></div>", unsafe_allow_html=True)

           strl.markdown("<br>", unsafe_allow_html=True)
           r1, r2, r3 = strl.columns(3)
           r1.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Margin Employed (approx.)</div><div class='kpi-value'>₹{agg['Total_Margin']:,.0f}</div></div>", unsafe_allow_html=True)
           r2.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Max Loss (worst case)</div><div class='kpi-value kpi-negative'>₹{agg['Total_Max_Loss']:,.0f}</div></div>", unsafe_allow_html=True)
           r3.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Max Profit (best case)</div><div class='kpi-value kpi-positive'>₹{agg['Total_Max_Profit']:,.0f}</div></div>", unsafe_allow_html=True)

           strl.markdown("---")
           strl.markdown("#### Open Positions")
           rows = []
           for idx, pos in enumerate(book):
               rows.append({
                   "#": idx, "Symbol": pos.symbol, "Strategy": pos.strategy.name, "Lots": pos.lots,
                   "Lot Size": pos.lot_size, "Max Profit": round(pos.strategy.max_profit * pos.lots * pos.lot_size, 2),
                   "Max Loss": round(pos.strategy.max_loss * pos.lots * pos.lot_size, 2),
                   "POP": round(pos.strategy.pop, 1), "Added": pos.added_on,
               })
           pos_df = pd.DataFrame(rows)
           strl.dataframe(pos_df, use_container_width=True, hide_index=True)

           rm_idx = strl.number_input("Position # to remove", min_value=0, max_value=max(0, len(book) - 1), value=0, step=1, key="rm_idx")
           rc1, rc2 = strl.columns(2)
           if rc1.button("🗑️ Remove Position", key="remove_position"):
               strl.session_state["portfolio"].pop(int(rm_idx))
               strl.rerun()
           if rc2.button("🧹 Clear Entire Book", key="clear_book"):
               strl.session_state["portfolio"] = []
               strl.rerun()

if __name__ == "__main__":
   try:
       main_app()
   except Exception as e:
       logger.error(f"Application crash: {e}")
       strl.error("Critical Engine Failure. Please review internal logs.")