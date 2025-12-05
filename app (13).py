"""
CVVC Crypto Index Rebalancer - COMPLETE VERSION
Features: Manual weights, Export reports, Clean UI, Visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

st.set_page_config(
    page_title="CVVC Index Rebalancer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CLEAN CSS - NO OVERLAPS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    body, p, span, div, label, li, td, th {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    .main {
        background-color: #ffffff !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-weight: 900 !important;
    }
    
    /* SIDEBAR - CLEAN */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    section[data-testid="stSidebar"] label {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
    }
    
    /* EXPANDER - NO OVERLAP */
    details summary {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-weight: 800 !important;
        padding: 1rem !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        list-style: none !important;
    }
    
    details summary::-webkit-details-marker {
        display: none !important;
    }
    
    details[open] summary {
        border-bottom: 2px solid #dee2e6 !important;
        border-radius: 8px 8px 0 0 !important;
    }
    
    /* BUTTON */
    button[kind="primary"] {
        background-color: #1e40af !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 900 !important;
        font-size: 1.2rem !important;
        padding: 1rem !important;
        width: 100% !important;
    }
    
    button[kind="secondary"] {
        background-color: #10b981 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* INPUTS */
    input {
        color: #000000 !important;
        font-weight: 600 !important;
        border: 2px solid #dee2e6 !important;
    }
    
    /* METRICS */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 3px solid #000000 !important;
        padding: 1rem !important;
        border-radius: 8px !important;
    }
    
    /* TABS */
    button[data-baseweb="tab"] {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1e40af !important;
        border-bottom: 3px solid #1e40af !important;
    }
</style>
""", unsafe_allow_html=True)

COLORS = ['#1e40af', '#10b981', '#f59e0b', '#8b5cf6']

# DATA LOADING
@st.cache_data
def load_data():
    import os
    
    # Try multiple possible locations for the data file
    possible_paths = [
        'ALL_TOKENS_weekly.csv',  # Same directory as script
        os.path.join(os.path.dirname(__file__), 'ALL_TOKENS_weekly.csv'),  # Script directory
        os.path.join(os.getcwd(), 'ALL_TOKENS_weekly.csv'),  # Current working directory
        '/mnt/user-data/outputs/ALL_TOKENS_weekly.csv',  # Absolute path (for local testing)
    ]
    
    # Try each path
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['market_cap'] > 0]
                df = df[df['close'] > 0]
                df = df.sort_values('date')
                st.success(f"✅ Data loaded from: {path}")
                return df
            except Exception as e:
                st.error(f"❌ Error reading {path}: {e}")
                continue
    
    # If we get here, file wasn't found anywhere
    st.error("❌ Could not find ALL_TOKENS_weekly.csv in any expected location!")
    st.error("**Searched in:**")
    for path in possible_paths:
        st.error(f"  - {path}")
    st.error(f"**Current directory:** {os.getcwd()}")
    st.error(f"**Files in current directory:** {os.listdir(os.getcwd())}")
    return None

# CCI30 BENCHMARK DATA
@st.cache_data
def load_cci30_data():
    """Load CCI30 index data"""
    try:
        df = pd.read_csv('cci30_OHLCV.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.rename(columns={'Date': 'date', 'Close': 'close'})
        df = df.sort_values('date')
        
        # Convert to weekly to match user data
        df = df.set_index('date')
        df_weekly = df['close'].resample('W-FRI').last().reset_index()
        df_weekly = df_weekly.dropna()
        
        return df_weekly
    except Exception as e:
        return pd.DataFrame()

def calculate_cci30_performance(cci30_data, start_date, end_date, initial):
    """Calculate CCI30 buy-and-hold performance"""
    cci = cci30_data[(cci30_data['date'] >= start_date) & (cci30_data['date'] <= end_date)].copy()
    
    if len(cci) == 0:
        return pd.DataFrame()
    
    # Calculate value based on initial investment (buy and hold)
    start_price = cci['close'].iloc[0]
    cci['value'] = initial * (cci['close'] / start_price)
    cci['rebalances'] = 0
    
    return cci[['date', 'value', 'rebalances']]

# CCI30 BENCHMARK DATA
@st.cache_data
def load_cci30_data():
    """Load CCI30 index data and convert to weekly"""
    import os
    
    # Try different possible filenames
    possible_names = ['cci30_OHLCV.csv', 'CCI30_OHLCV.csv', 'cci30_ohlcv.csv']
    
    for filename in possible_names:
        try:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.rename(columns={'Date': 'date', 'Close': 'close'})
                df = df.sort_values('date')
                
                # Convert to weekly (match user data frequency)
                df = df.set_index('date')
                df_weekly = df['close'].resample('W-FRI').last().reset_index()
                df_weekly = df_weekly.dropna()
                
                return df_weekly
        except Exception as e:
            continue
    
    # If we get here, file not found with any name
    return pd.DataFrame()

@st.cache_data
def load_cci30_weights():
    """Load CCI30 monthly weights and convert to format matching strategy weights"""
    import os
    
    # Mapping from CCI30 coin names to common symbols
    coin_to_symbol = {
        'bitcoin': 'BTC',
        'ethereum': 'ETH',
        'binancecoin': 'BNB',
        'bnb': 'BNB',
        'xrp': 'XRP',
        'solana': 'SOL',
        'cardano': 'ADA',
        'dogecoin': 'DOGE',
        'tron': 'TRX',
        'avalanche': 'AVAX',
        'chainlink': 'LINK',
        'polkadot-new': 'DOT',
        'polkadot': 'DOT',
        'polygon': 'MATIC',
        'matic': 'MATIC',
        'litecoin': 'LTC',
        'bitcoin-cash': 'BCH',
        'stellar': 'XLM',
        'monero': 'XMR',
        'uniswap': 'UNI',
        'shiba-inu': 'SHIB',
        'toncoin': 'TON',
        'hedera': 'HBAR',
        'cronos': 'CRO',
        'sui': 'SUI',
        'aave': 'AAVE',
        'pepe': 'PEPE'
    }
    
    possible_names = ['cci30_monthly_weights.csv', 'CCI30_monthly_weights.csv']
    
    for filename in possible_names:
        try:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Clean weight column (remove % and convert to float)
                df['Weight'] = df['Weight in Index'].str.rstrip('%').astype(float) / 100.0
                
                # Map coin names to symbols
                df['Symbol'] = df['Coin'].map(coin_to_symbol)
                df = df[df['Symbol'].notna()]  # Keep only mapped coins
                
                # Pivot to get dates as rows, symbols as columns
                weights_pivot = df.pivot_table(
                    index='Date',
                    columns='Symbol',
                    values='Weight',
                    aggfunc='first'
                ).reset_index()
                
                weights_pivot = weights_pivot.rename(columns={'Date': 'date'})
                weights_pivot = weights_pivot.fillna(0)
                
                return weights_pivot
        except Exception as e:
            continue
    
    return pd.DataFrame()

def calculate_cci30_performance(cci30_data, start_date, end_date, initial):
    """Calculate CCI30 buy-and-hold performance"""
    # Filter by date range
    cci = cci30_data[(cci30_data['date'] >= start_date) & (cci30_data['date'] <= end_date)].copy()
    
    if len(cci) == 0:
        return pd.DataFrame()
    
    # Calculate value based on initial investment (buy and hold)
    start_price = cci['close'].iloc[0]
    cci['value'] = initial * (cci['close'] / start_price)
    cci['rebalances'] = 0
    
    return cci[['date', 'value', 'rebalances']]

# EXPORT FUNCTIONS
def create_performance_csv(result, initial):
    perf = result['performance'].copy()
    perf['return_pct'] = ((perf['value'] / initial) - 1) * 100
    return perf

def create_weights_csv(result):
    if result['weights'].empty:
        return None
    return result['weights'].copy()

def create_report_txt(result, initial):
    final = result['performance']['value'].iloc[-1]
    m = result['metrics']
    c = result['config']
    
    report = f"""
═══════════════════════════════════════════════════════════════
                    {result['name']}
═══════════════════════════════════════════════════════════════

INVESTMENT:
Initial:              ${initial:,.2f}
Final:                ${final:,.2f}

PERFORMANCE:
Total Return:         {m['Total Return (%)']}%
Annual Return:        {m['Annual Return (%)']}%
Volatility:           {m['Volatility (%)']}%
Sharpe Ratio:         {m['Sharpe Ratio']}
Max Drawdown:         {m['Max Drawdown (%)']}%
Calmar Ratio:         {m['Calmar Ratio']}
Rebalances:           {m['Rebalances']}

CONFIGURATION:
Method:               {c['method'].replace('_', ' ').title()}
Frequency:            {c['rebalance_freq'].title()}
Top N Tokens:         {c.get('top_n', 'N/A')}
"""
    
    # Only add cap settings for non-benchmark strategies
    if c.get('method') != 'benchmark':
        report += f"""BTC Cap:              {c.get('btc_cap', 1.0)*100:.0f}%
ETH Cap:              {c.get('eth_cap', 1.0)*100:.0f}%
BTC+ETH Combined:     {c.get('btc_eth_combined', 1.0)*100:.0f}%
Max Any Token:        {c.get('max_any', 1.0)*100:.0f}%
Min Weight:           {c.get('min_weight', 0.0)*100:.1f}%
"""
    
    if c.get('method') == 'manual' and c.get('manual_weights'):
        report += "\nMANUAL WEIGHTS:\n"
        for token, weight in sorted(c['manual_weights'].items(), key=lambda x: x[1], reverse=True):
            report += f"{token:6s}  {weight:6.2f}%\n"
    
    return report

# INDEX STRATEGY CLASS
class IndexStrategy:
    def __init__(self, config):
        self.name = config['name']
        self.rebalance_freq = config['rebalance_freq']
        self.top_n = config['top_n']
        self.btc_cap = config.get('btc_cap', 1.0)
        self.eth_cap = config.get('eth_cap', 1.0)
        self.btc_eth_combined = config.get('btc_eth_combined', 1.0)
        self.max_any = config.get('max_any', 1.0)
        self.excluded = config.get('excluded', [])
        self.min_weight = config.get('min_weight', 0.0)
        self.method = config.get('method', 'market_cap')
        self.manual_weights = config.get('manual_weights', {})
        self.ewma_alpha = config.get('ewma_alpha', 0.1)
        self.ewma_lookback = config.get('ewma_lookback', 13)
    
    def calculate_weights(self, data, date):
        snapshot = data[data['date'] == date].copy()
        if len(snapshot) == 0:
            return {}
        
        if self.excluded:
            snapshot = snapshot[~snapshot['symbol'].isin(self.excluded)]
        
        snapshot = snapshot.nlargest(self.top_n, 'market_cap')
        if len(snapshot) == 0:
            return {}
        
        # Base weights by method
        if self.method == 'manual':
            weights = self._manual_weights(snapshot)
        elif self.method == 'equal':
            weights = self._equal_weights(snapshot)
        elif self.method == 'ewma':
            weights = self._ewma_weights(data, date, snapshot)
        else:
            weights = self._market_cap_weights(snapshot)
        
        if not weights:
            return {}
        
        # Apply caps
        weights = self._apply_caps(weights)
        
        # Filter minimum
        if self.min_weight > 0:
            weights = {k: v for k, v in weights.items() if v >= self.min_weight}
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _market_cap_weights(self, snapshot):
        total = snapshot['market_cap'].sum()
        if total == 0:
            return {}
        return {row['symbol']: row['market_cap'] / total for _, row in snapshot.iterrows()}
    
    def _equal_weights(self, snapshot):
        n = len(snapshot)
        if n == 0:
            return {}
        weight = 1.0 / n
        return {row['symbol']: weight for _, row in snapshot.iterrows()}
    
    def _manual_weights(self, snapshot):
        weights = {}
        available = set(snapshot['symbol'].values)
        
        for symbol in available:
            if symbol in self.manual_weights:
                weights[symbol] = self.manual_weights[symbol] / 100.0
            else:
                weights[symbol] = 0.0
        
        if sum(weights.values()) == 0:
            return self._market_cap_weights(snapshot)
        
        return weights
    
    def _ewma_weights(self, data, date, snapshot):
        """Calculate weights using Exponentially Weighted Moving Average (CCI30-style)"""
        ewma_mcaps = {}
        
        # For each token in snapshot, calculate EWMA of historical market caps
        for _, row in snapshot.iterrows():
            symbol = row['symbol']
            
            # Get historical data up to current date
            hist = data[(data['symbol'] == symbol) & (data['date'] <= date)].copy()
            hist = hist.sort_values('date', ascending=False)  # Most recent first
            
            # Limit to lookback window
            hist = hist.head(self.ewma_lookback)
            
            if len(hist) == 0:
                continue
            
            # Calculate EWMA: M*(T) = Σ[M(T-i) × exp(-αi)] / Σ[exp(-αi)]
            numerator = 0
            denominator = 0
            
            for i, mcap in enumerate(hist['market_cap'].values):
                weight = np.exp(-self.ewma_alpha * i)
                numerator += mcap * weight
                denominator += weight
            
            if denominator > 0:
                ewma_mcaps[symbol] = numerator / denominator
        
        if not ewma_mcaps:
            return {}
        
        # Apply square root to EWMA market caps (CCI30 methodology)
        sqrt_ewma = {token: np.sqrt(mcap) for token, mcap in ewma_mcaps.items()}
        total_sqrt = sum(sqrt_ewma.values())
        
        if total_sqrt == 0:
            return {}
        
        # Calculate weights
        weights = {token: sqrt_val / total_sqrt for token, sqrt_val in sqrt_ewma.items()}
        
        return weights
    
    def _apply_caps(self, weights):
        """Apply all caps with smart redistribution"""
        
        for iteration in range(50):  # More iterations for complex cases
            old_weights = weights.copy()
            changed = False
            
            # Apply BTC+ETH combined cap
            btc = weights.get('BTC', 0)
            eth = weights.get('ETH', 0)
            if btc + eth > self.btc_eth_combined:
                # Reduce proportionally
                total = btc + eth
                if total > 0:
                    weights['BTC'] = self.btc_eth_combined * (btc / total)
                    weights['ETH'] = self.btc_eth_combined * (eth / total)
                    changed = True
            
            # Apply individual BTC cap
            if 'BTC' in weights and weights['BTC'] > self.btc_cap:
                weights['BTC'] = self.btc_cap
                changed = True
            
            # Apply individual ETH cap
            if 'ETH' in weights and weights['ETH'] > self.eth_cap:
                weights['ETH'] = self.eth_cap
                changed = True
            
            # Apply max_any cap to all tokens EXCEPT BTC and ETH
            if self.max_any < 1.0:
                for token in weights:
                    # Skip BTC and ETH - they have their own caps
                    if token not in ['BTC', 'ETH']:
                        if weights[token] > self.max_any:
                            weights[token] = self.max_any
                            changed = True
            
            # Normalize to 100% (this redistributes excess to uncapped tokens)
            total = sum(weights.values())
            if total > 0 and abs(total - 1.0) > 0.0001:
                # Only redistribute to tokens that aren't at their cap
                uncapped_tokens = {}
                for token, weight in weights.items():
                    at_cap = False
                    if token == 'BTC' and weight >= self.btc_cap - 0.0001:
                        at_cap = True
                    if token == 'ETH' and weight >= self.eth_cap - 0.0001:
                        at_cap = True
                    # Max any cap only applies to non-BTC/ETH tokens
                    if token not in ['BTC', 'ETH'] and self.max_any < 1.0 and weight >= self.max_any - 0.0001:
                        at_cap = True
                    
                    if not at_cap:
                        uncapped_tokens[token] = weight
                
                # If there are uncapped tokens, redistribute to them proportionally
                if uncapped_tokens and sum(uncapped_tokens.values()) > 0:
                    uncapped_sum = sum(uncapped_tokens.values())
                    deficit = 1.0 - total
                    
                    for token in weights:
                        if token in uncapped_tokens:
                            # Add proportional share of deficit
                            weights[token] += (uncapped_tokens[token] / uncapped_sum) * deficit
                else:
                    # All tokens at cap, just normalize
                    weights = {k: v/total for k, v in weights.items()}
            
            # Check convergence
            if not changed and self._weights_equal(weights, old_weights):
                break
        
        # Ensure weights sum to exactly 100% without violating caps
        # Only adjust if sum is significantly off (> 0.01%)
        total = sum(weights.values())
        if abs(total - 1.0) > 0.0001:
            # Find uncapped tokens for final adjustment
            uncapped = {}
            for token, weight in weights.items():
                at_cap = False
                if token == 'BTC' and weight >= self.btc_cap - 0.0001:
                    at_cap = True
                if token == 'ETH' and weight >= self.eth_cap - 0.0001:
                    at_cap = True
                # Max any cap only applies to non-BTC/ETH tokens
                if token not in ['BTC', 'ETH'] and self.max_any < 1.0 and weight >= self.max_any - 0.0001:
                    at_cap = True
                if not at_cap:
                    uncapped[token] = weight
            
            # Only adjust uncapped tokens
            if uncapped and sum(uncapped.values()) > 0:
                deficit = 1.0 - total
                uncapped_sum = sum(uncapped.values())
                for token in weights:
                    if token in uncapped:
                        weights[token] += (uncapped[token] / uncapped_sum) * deficit
        
        return weights
    
    def _weights_equal(self, w1, w2, tolerance=0.0001):
        """Check if two weight dicts are approximately equal"""
        if set(w1.keys()) != set(w2.keys()):
            return False
        for k in w1:
            if abs(w1[k] - w2.get(k, 0)) > tolerance:
                return False
        return True
    
    def _cap_btc(self, weights):
        """Cap BTC to btc_cap"""
        if 'BTC' in weights and weights['BTC'] > self.btc_cap:
            weights['BTC'] = self.btc_cap
        return weights
    
    def _cap_eth(self, weights):
        """Cap ETH to eth_cap"""
        if 'ETH' in weights and weights['ETH'] > self.eth_cap:
            weights['ETH'] = self.eth_cap
        return weights
    
    def _cap_btc_eth_combined(self, weights):
        """Cap BTC+ETH combined to btc_eth_combined"""
        btc = weights.get('BTC', 0)
        eth = weights.get('ETH', 0)
        combined = btc + eth
        
        if combined > self.btc_eth_combined:
            # Reduce BTC and ETH proportionally
            if combined > 0:
                weights['BTC'] = self.btc_eth_combined * (btc / combined)
                weights['ETH'] = self.btc_eth_combined * (eth / combined)
        return weights
    
    def _cap_any_token(self, weights):
        """Cap any token to max_any"""
        if self.max_any >= 1.0:
            return weights
        
        # Cap all tokens above max_any
        for token in weights:
            if weights[token] > self.max_any:
                weights[token] = self.max_any
        
        return weights
    
    # BACKTEST FUNCTION

# BACKTESTING
def get_rebalance_dates(dates, freq):
    freq_map = {'weekly': 1, 'monthly': 4, 'quarterly': 13, 'semiannually': 26}
    step = freq_map.get(freq, 13)
    
    if len(dates) == 0:
        return []
    
    rebalance_dates = [dates[0]]
    for i in range(step, len(dates), step):
        rebalance_dates.append(dates[i])
    
    return rebalance_dates

def backtest_strategy(strategy, data, start_date, end_date, initial=10000):
    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)].copy()
    
    if len(data) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    dates = sorted(data['date'].unique())
    if len(dates) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    rebalance_dates = get_rebalance_dates(dates, strategy.rebalance_freq)
    
    portfolio_value = initial
    holdings = {}
    values = []
    weights_history = []
    rebalances = 0
    last_known_prices = {}  # Track last known price for each token
    
    for date in dates:
        # Update last known prices for all tokens with data on this date
        date_data = data[data['date'] == date]
        for _, row in date_data.iterrows():
            if row['close'] > 0:
                last_known_prices[row['symbol']] = row['close']
        
        if date in rebalance_dates:
            weights = strategy.calculate_weights(data, date)
            
            if not weights:
                continue
            
            holdings = {}
            for symbol, weight in weights.items():
                # Only buy tokens that have price data on THIS date
                price = data[(data['date'] == date) & (data['symbol'] == symbol)]['close'].values
                if len(price) > 0 and price[0] > 0:
                    holdings[symbol] = (portfolio_value * weight) / price[0]
                    last_known_prices[symbol] = price[0]
            
            rebalances += 1
            weight_record = {'date': date}
            weight_record.update(weights)
            weights_history.append(weight_record)
        
        # Calculate portfolio value using last known prices
        new_value = 0
        for symbol, quantity in holdings.items():
            if symbol in last_known_prices:
                new_value += quantity * last_known_prices[symbol]
        
        if new_value > 0:
            portfolio_value = new_value
        
        values.append({'date': date, 'value': portfolio_value})
    
    if len(values) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    performance = pd.DataFrame(values)
    performance['rebalances'] = rebalances
    
    weights_df = pd.DataFrame(weights_history) if weights_history else pd.DataFrame()
    if not weights_df.empty:
        weights_df = weights_df.fillna(0)
    
    return performance, weights_df

def calculate_metrics(perf, initial=10000):
    if perf.empty or len(perf) < 2:
        return {
            'Total Return (%)': 0.0,
            'Annual Return (%)': 0.0,
            'Volatility (%)': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown (%)': 0.0,
            'Calmar Ratio': 0.0,
            'Rebalances': 0
        }
    
    values = perf['value'].values
    returns = np.diff(values) / values[:-1]
    returns = returns[np.isfinite(returns)]
    
    total_return = (values[-1] - initial) / initial
    
    weeks = len(values)
    years = weeks / 52.0
    
    if years > 0 and total_return > -1:
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = 0.0
    
    volatility = np.std(returns) * np.sqrt(52) if len(returns) > 0 else 0.0
    sharpe = annual_return / volatility if volatility > 0 else 0.0
    
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    drawdown = drawdown[np.isfinite(drawdown)]
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
    
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0
    rebalances = int(perf['rebalances'].iloc[0]) if 'rebalances' in perf.columns else 0
    
    return {
        'Total Return (%)': round(total_return * 100, 2),
        'Annual Return (%)': round(annual_return * 100, 2),
        'Volatility (%)': round(volatility * 100, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Max Drawdown (%)': round(max_drawdown * 100, 2),
        'Calmar Ratio': round(calmar, 2),
        'Rebalances': rebalances
    }

# MAIN APP
def main():
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); 
                padding: 2rem; border-radius: 12px; margin-bottom: 2rem; text-align: center;'>
        <h1 style='margin: 0; color: #ffffff; font-size: 2.5rem; font-weight: 900;'>
            📊 CVVC Index Rebalancer
        </h1>
        <p style='margin: 0.5rem 0 0 0; color: #ffffff; font-size: 1.1rem; font-weight: 700;'>
            Market Cap · Equal Weight · EWMA · Manual Allocation · Export Reports
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    data = load_data()
    if data is None:
        st.stop()
    
    # Load CCI30 benchmark data
    cci30_data = load_cci30_data()
    cci30_weights_data = load_cci30_weights()
    has_cci30 = not cci30_data.empty
    
    all_tokens = sorted(data['symbol'].unique())
    min_date = data['date'].min()
    max_date = data['date'].max()
    
    # SIDEBAR
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")
    
    # Date Range
    st.sidebar.subheader("📅 Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date)
    
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    if start_date >= end_date:
        st.sidebar.error("⚠️ Invalid range!")
    
    st.sidebar.markdown("---")
    
    # Investment
    st.sidebar.subheader("💰 Investment")
    initial_investment = st.sidebar.number_input(
        "Starting Amount ($)",
        min_value=1000,
        max_value=10000000,
        value=10000,
        step=1000
    )
    st.sidebar.success(f"**${initial_investment:,.0f}**")
    
    st.sidebar.markdown("---")
    
    # Exclusions
    st.sidebar.subheader("🚫 Exclude Tokens")
    excluded = st.sidebar.multiselect(
        "Remove tokens:",
        options=all_tokens,
        default=[]
    )
    
    st.sidebar.markdown("---")
    
    # CCI30 Benchmark
    st.sidebar.subheader("📊 CCI30 Benchmark")
    if has_cci30:
        show_cci30 = st.sidebar.checkbox(
            "Show CCI30 for comparison",
            value=False,
            help="Compare your strategies against the CCI30 Index (buy-and-hold)"
        )
        if not show_cci30:
            st.sidebar.info("💡 Enable to compare vs CCI30")
    else:
        st.sidebar.warning("⚠️ CCI30 data not available")
        # Debug: show what files we can see
        import os
        files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if files:
            st.sidebar.caption(f"Files found: {', '.join(files)}")
        show_cci30 = False
    
    st.sidebar.markdown("---")
    st.sidebar.title("🎯 Strategies")
    
    strategies = []
    
    # STRATEGY 1
    st.sidebar.markdown("### Strategy 1")
    s1_enabled = st.sidebar.checkbox("Enable Strategy 1", value=True, key="s1_enable")
    
    if s1_enabled:
        with st.sidebar.expander("Configure Strategy 1"):
            s1_name = st.text_input("Name", "Baseline", key="s1_name")
            
            col1, col2 = st.columns(2)
            with col1:
                s1_freq = st.selectbox("Frequency", ['weekly', 'monthly', 'quarterly', 'semiannually'], index=2, key="s1_freq")
            with col2:
                s1_top = st.slider("Top N", 5, 27, 20, key="s1_top")
            
            s1_method = st.radio(
                "Weighting",
                ['market_cap', 'equal', 'ewma', 'manual'],
                index=0,
                key="s1_method",
                format_func=lambda x: {'market_cap': 'Market Cap', 'equal': 'Equal', 'ewma': 'EWMA', 'manual': 'Manual'}[x]
            )
            
            s1_manual = {}
            if s1_method == 'manual':
                st.markdown("**Enter Weights (%):**")
                # Use ALL tokens from dataset
                manual_tokens = sorted(all_tokens)  # All 24 tokens alphabetically
                for token in manual_tokens:
                    w = st.number_input(f"{token}", 0.0, 100.0, 0.0, 1.0, key=f"s1_{token}")
                    if w > 0:
                        s1_manual[token] = w
            
            # EWMA parameters
            s1_ewma_alpha = 0.1
            s1_ewma_lookback = 13
            if s1_method == 'ewma':
                st.markdown("**EWMA Parameters:**")
                st.info("📘 Uses CCI30 methodology: Exponentially Weighted Moving Average + Square Root weighting")
                s1_ewma_alpha = st.slider("Decay α", 0.01, 0.30, 0.10, 0.01, key="s1_ewma_alpha", 
                                          help="Lower = more history, Higher = more recent focus")
                s1_ewma_lookback = st.slider("Lookback (weeks)", 4, 26, 13, 1, key="s1_ewma_lookback",
                                             help="Number of weeks to include in calculation")
            
            st.markdown("**Caps:**")
            s1_btc = st.slider("BTC %", 0, 100, 100, 5, key="s1_btc") / 100
            s1_eth = st.slider("ETH %", 0, 100, 100, 5, key="s1_eth") / 100
            s1_be = st.slider("BTC+ETH %", 0, 100, 100, 5, key="s1_be") / 100
            s1_max = st.slider("Max Any %", 5, 100, 100, 5, key="s1_max") / 100
            s1_min = st.slider("Min %", 0.0, 5.0, 0.0, 0.1, key="s1_min") / 100
            
            strategies.append({
                'name': s1_name, 'rebalance_freq': s1_freq, 'top_n': s1_top,
                'method': s1_method, 'manual_weights': s1_manual,
                'ewma_alpha': s1_ewma_alpha, 'ewma_lookback': s1_ewma_lookback,
                'btc_cap': s1_btc, 'eth_cap': s1_eth, 'btc_eth_combined': s1_be,
                'max_any': s1_max, 'excluded': excluded, 'min_weight': s1_min
            })
    
    # STRATEGY 2
    st.sidebar.markdown("### Strategy 2")
    s2_enabled = st.sidebar.checkbox("Enable Strategy 2", value=True, key="s2_enable")
    
    if s2_enabled:
        with st.sidebar.expander("Configure Strategy 2"):
            s2_name = st.text_input("Name", "Capped", key="s2_name")
            
            col1, col2 = st.columns(2)
            with col1:
                s2_freq = st.selectbox("Frequency", ['weekly', 'monthly', 'quarterly', 'semiannually'], index=2, key="s2_freq")
            with col2:
                s2_top = st.slider("Top N", 5, 27, 20, key="s2_top")
            
            s2_method = st.radio(
                "Weighting",
                ['market_cap', 'equal', 'ewma', 'manual'],
                index=0,
                key="s2_method",
                format_func=lambda x: {'market_cap': 'Market Cap', 'equal': 'Equal', 'ewma': 'EWMA', 'manual': 'Manual'}[x]
            )
            
            s2_manual = {}
            if s2_method == 'manual':
                st.markdown("**Enter Weights (%):**")
                # Use ALL tokens from dataset
                manual_tokens = sorted(all_tokens)  # All 24 tokens alphabetically
                for token in manual_tokens:
                    w = st.number_input(f"{token}", 0.0, 100.0, 0.0, 1.0, key=f"s2_{token}")
                    if w > 0:
                        s2_manual[token] = w
            
            # EWMA parameters
            s2_ewma_alpha = 0.1
            s2_ewma_lookback = 13
            if s2_method == 'ewma':
                st.markdown("**EWMA Parameters:**")
                st.info("📘 Uses CCI30 methodology: Exponentially Weighted Moving Average + Square Root weighting")
                s2_ewma_alpha = st.slider("Decay α", 0.01, 0.30, 0.10, 0.01, key="s2_ewma_alpha",
                                          help="Lower = more history, Higher = more recent focus")
                s2_ewma_lookback = st.slider("Lookback (weeks)", 4, 26, 13, 1, key="s2_ewma_lookback",
                                             help="Number of weeks to include in calculation")
            
            st.markdown("**Caps:**")
            s2_btc = st.slider("BTC %", 0, 100, 30, 5, key="s2_btc") / 100
            s2_eth = st.slider("ETH %", 0, 100, 20, 5, key="s2_eth") / 100
            s2_be = st.slider("BTC+ETH %", 0, 100, 50, 5, key="s2_be") / 100
            s2_max = st.slider("Max Any %", 5, 100, 100, 5, key="s2_max") / 100
            s2_min = st.slider("Min %", 0.0, 5.0, 0.0, 0.1, key="s2_min") / 100
            
            strategies.append({
                'name': s2_name, 'rebalance_freq': s2_freq, 'top_n': s2_top,
                'method': s2_method, 'manual_weights': s2_manual,
                'ewma_alpha': s2_ewma_alpha, 'ewma_lookback': s2_ewma_lookback,
                'btc_cap': s2_btc, 'eth_cap': s2_eth, 'btc_eth_combined': s2_be,
                'max_any': s2_max, 'excluded': excluded, 'min_weight': s2_min
            })
    
    # STRATEGY 3
    st.sidebar.markdown("### Strategy 3")
    s3_enabled = st.sidebar.checkbox("Enable Strategy 3", value=False, key="s3_enable")
    
    if s3_enabled:
        with st.sidebar.expander("Configure Strategy 3"):
            s3_name = st.text_input("Name", "Custom", key="s3_name")
            
            col1, col2 = st.columns(2)
            with col1:
                s3_freq = st.selectbox("Frequency", ['weekly', 'monthly', 'quarterly', 'semiannually'], index=2, key="s3_freq")
            with col2:
                s3_top = st.slider("Top N", 5, 27, 20, key="s3_top")
            
            s3_method = st.radio(
                "Weighting",
                ['market_cap', 'equal', 'ewma', 'manual'],
                index=0,
                key="s3_method",
                format_func=lambda x: {'market_cap': 'Market Cap', 'equal': 'Equal', 'ewma': 'EWMA', 'manual': 'Manual'}[x]
            )
            
            s3_manual = {}
            if s3_method == 'manual':
                st.markdown("**Enter Weights (%):**")
                # Use ALL tokens from dataset
                manual_tokens = sorted(all_tokens)  # All 24 tokens alphabetically
                for token in manual_tokens:
                    w = st.number_input(f"{token}", 0.0, 100.0, 0.0, 1.0, key=f"s3_{token}")
                    if w > 0:
                        s3_manual[token] = w
            
            # EWMA parameters
            s3_ewma_alpha = 0.1
            s3_ewma_lookback = 13
            if s3_method == 'ewma':
                st.markdown("**EWMA Parameters:**")
                st.info("📘 Uses CCI30 methodology: Exponentially Weighted Moving Average + Square Root weighting")
                s3_ewma_alpha = st.slider("Decay α", 0.01, 0.30, 0.10, 0.01, key="s3_ewma_alpha",
                                          help="Lower = more history, Higher = more recent focus")
                s3_ewma_lookback = st.slider("Lookback (weeks)", 4, 26, 13, 1, key="s3_ewma_lookback",
                                             help="Number of weeks to include in calculation")
            
            st.markdown("**Caps:**")
            s3_btc = st.slider("BTC %", 0, 100, 25, 5, key="s3_btc") / 100
            s3_eth = st.slider("ETH %", 0, 100, 15, 5, key="s3_eth") / 100
            s3_be = st.slider("BTC+ETH %", 0, 100, 40, 5, key="s3_be") / 100
            s3_max = st.slider("Max Any %", 5, 100, 15, 5, key="s3_max") / 100
            s3_min = st.slider("Min %", 0.0, 5.0, 1.0, 0.1, key="s3_min") / 100
            
            strategies.append({
                'name': s3_name, 'rebalance_freq': s3_freq, 'top_n': s3_top,
                'method': s3_method, 'manual_weights': s3_manual,
                'ewma_alpha': s3_ewma_alpha, 'ewma_lookback': s3_ewma_lookback,
                'btc_cap': s3_btc, 'eth_cap': s3_eth, 'btc_eth_combined': s3_be,
                'max_any': s3_max, 'excluded': excluded, 'min_weight': s3_min
            })
    
    st.sidebar.markdown("---")
    
    # RUN BUTTON
    if st.sidebar.button("🚀 RUN BACKTEST", type="primary"):
        if not strategies:
            st.sidebar.error("⚠️ Enable a strategy!")
        elif start_date >= end_date:
            st.sidebar.error("⚠️ Invalid date range!")
        else:
            with st.spinner('⏳ Running...'):
                results = []
                progress = st.sidebar.progress(0)
                
                for i, cfg in enumerate(strategies):
                    strategy = IndexStrategy(cfg)
                    perf, weights = backtest_strategy(strategy, data, start_date, end_date, initial_investment)
                    
                    if not perf.empty:
                        metrics = calculate_metrics(perf, initial_investment)
                        results.append({
                            'name': cfg['name'],
                            'config': cfg,
                            'performance': perf,
                            'weights': weights,
                            'metrics': metrics,
                            'color': COLORS[i % len(COLORS)]
                        })
                    
                    progress.progress((i + 1) / len(strategies))
                
                # Add CCI30 benchmark if enabled
                if show_cci30 and has_cci30:
                    cci30_perf = calculate_cci30_performance(cci30_data, start_date, end_date, initial_investment)
                    if not cci30_perf.empty:
                        cci30_metrics = calculate_metrics(cci30_perf, initial_investment)
                        
                        # Filter CCI30 weights by date range
                        cci30_weights_filtered = pd.DataFrame()
                        if not cci30_weights_data.empty:
                            cci30_weights_filtered = cci30_weights_data[
                                (cci30_weights_data['date'] >= start_date) & 
                                (cci30_weights_data['date'] <= end_date)
                            ].copy()
                        
                        results.append({
                            'name': 'CCI30 Index',
                            'config': {'method': 'benchmark', 'rebalance_freq': 'buy-hold', 'top_n': 30},
                            'performance': cci30_perf,
                            'weights': cci30_weights_filtered,
                            'metrics': cci30_metrics,
                            'color': '#94a3b8'  # Gray for benchmark
                        })
                
                if results:
                    st.session_state['results'] = results
                    st.session_state['initial_investment'] = initial_investment
                    progress.empty()
                    st.sidebar.success(f"✅ {len(results)} done!")
                else:
                    st.sidebar.error("❌ No results")
    
    # RESULTS
    if 'results' in st.session_state and st.session_state['results']:
        results = st.session_state['results']
        initial_inv = st.session_state.get('initial_investment', initial_investment)
        
        # SUMMARY CARDS
        st.markdown("## 📊 Performance Summary")
        
        cols = st.columns(len(results))
        for i, r in enumerate(results):
            with cols[i]:
                ret = r['metrics']['Total Return (%)']
                final = initial_inv * (1 + ret/100)
                
                st.markdown(f"""
                <div style='background: white; padding: 1.5rem; border: 3px solid {r['color']}; 
                            border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                    <h3 style='margin: 0 0 1rem 0; color: #000000;'>{r['name']}</h3>
                    <p style='margin: 0; font-size: 0.9rem;'>${initial_inv:,.0f} → ${final:,.0f}</p>
                    <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem; font-weight: 700;'>Total Return</p>
                    <p style='margin: 0; font-size: 2rem; font-weight: 900; color: {'#10b981' if ret > 0 else '#ef4444'};'>
                        {'+' if ret > 0 else ''}{ret}%
                    </p>
                    <p style='margin: 1rem 0 0 0; font-size: 0.85rem;'><strong>Sharpe:</strong> {r['metrics']['Sharpe Ratio']}</p>
                    <p style='margin: 0.25rem 0 0 0; font-size: 0.85rem;'><strong>Rebalances:</strong> {r['metrics']['Rebalances']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # DOWNLOAD SECTION
        st.markdown("## 📥 Download Reports")
        
        cols = st.columns(len(results))
        for i, r in enumerate(results):
            with cols[i]:
                st.markdown(f"**{r['name']}**")
                
                # Performance CSV
                perf_csv = create_performance_csv(r, initial_inv)
                csv_buf = io.StringIO()
                perf_csv.to_csv(csv_buf, index=False)
                
                st.download_button(
                    "📊 Performance CSV",
                    data=csv_buf.getvalue(),
                    file_name=f"{r['name']}_performance.csv",
                    mime="text/csv",
                    key=f"perf_{i}"
                )
                
                # Weights CSV
                weights_csv = create_weights_csv(r)
                if weights_csv is not None:
                    weights_buf = io.StringIO()
                    weights_csv.to_csv(weights_buf, index=False)
                    
                    st.download_button(
                        "⚖️ Weights CSV",
                        data=weights_buf.getvalue(),
                        file_name=f"{r['name']}_weights.csv",
                        mime="text/csv",
                        key=f"weights_{i}"
                    )
                
                # Report TXT
                report_txt = create_report_txt(r, initial_inv)
                
                st.download_button(
                    "📄 Report TXT",
                    data=report_txt,
                    file_name=f"{r['name']}_report.txt",
                    mime="text/plain",
                    key=f"report_{i}"
                )
        
        st.markdown("---")
        
        # TABS
        tab1, tab2, tab3, tab4 = st.tabs(["📈 PORTFOLIO VALUE", "⚖️ WEIGHTS", "📊 METRICS", "⚙️ CONFIG"])
        
        with tab1:
            st.markdown("### Portfolio Value Over Time")
            
            fig = go.Figure()
            for r in results:
                # Use dashed line for CCI30, solid for strategies
                dash_style = 'dash' if r['name'] == 'CCI30 Index' else None
                line_width = 3 if r['name'] == 'CCI30 Index' else 4
                
                fig.add_trace(go.Scatter(
                    x=r['performance']['date'],
                    y=r['performance']['value'],
                    name=r['name'],
                    line=dict(color=r['color'], width=line_width, dash=dash_style),
                    mode='lines'
                ))
            
            fig.update_layout(
                height=600,
                hovermode='x unified',
                template='simple_white',
                font=dict(size=14, color='#000000'),
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                legend=dict(font=dict(size=14))
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                best = max(results, key=lambda x: x['metrics']['Total Return (%)'])
                st.metric("🏆 Best Return", best['name'], f"{best['metrics']['Total Return (%)']}%")
            with col2:
                best = max(results, key=lambda x: x['metrics']['Sharpe Ratio'])
                st.metric("⭐ Best Sharpe", best['name'], f"{best['metrics']['Sharpe Ratio']}")
            with col3:
                best = min(results, key=lambda x: x['metrics']['Max Drawdown (%)'])
                st.metric("🛡️ Min Drawdown", best['name'], f"{best['metrics']['Max Drawdown (%)']}%")
            with col4:
                best = max(results, key=lambda x: x['metrics']['Calmar Ratio'])
                st.metric("🎯 Best Calmar", best['name'], f"{best['metrics']['Calmar Ratio']}")
        
        with tab2:
            st.markdown("### Token Weights Over Time")
            
            selected = st.selectbox("Select strategy:", [r['name'] for r in results])
            result = next((r for r in results if r['name'] == selected), results[0])
            
            if not result['weights'].empty:
                weights_df = result['weights']
                
                weight_cols = [c for c in weights_df.columns if c != 'date']
                avg_weights = weights_df[weight_cols].mean().sort_values(ascending=False)
                top_tokens = avg_weights.head(10).index.tolist()
                
                # Stacked area chart
                fig = go.Figure()
                for token in top_tokens:
                    if token in weights_df.columns:
                        fig.add_trace(go.Scatter(
                            x=weights_df['date'],
                            y=weights_df[token] * 100,
                            name=token,
                            mode='lines',
                            stackgroup='one',
                            hovertemplate='%{y:.2f}%'
                        ))
                
                fig.update_layout(
                    height=600,
                    title=f"Top 10 Tokens - {selected}",
                    hovermode='x unified',
                    template='simple_white',
                    xaxis_title="Date",
                    yaxis_title="Weight (%)",
                    legend=dict(font=dict(size=12))
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Latest weights table
                st.markdown("#### Latest Allocation")
                latest = weights_df.iloc[-1].drop('date').sort_values(ascending=False)
                latest = latest[latest > 0]
                
                weight_table = pd.DataFrame({
                    'Token': latest.index,
                    'Weight (%)': [f"{v*100:.2f}%" for v in latest.values],
                    'Weight': latest.values
                })
                
                # Bar chart
                fig = px.bar(
                    weight_table.head(15),
                    x='Token',
                    y='Weight',
                    title="Current Top 15 Holdings",
                    color='Weight',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis_title="Weight (%)",
                    yaxis_tickformat='.0%'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.dataframe(
                    weight_table[['Token', 'Weight (%)']],
                    use_container_width=True,
                    height=400
                )
            else:
                st.warning("No weight data available")
        
        with tab3:
            st.markdown("### Full Metrics Comparison")
            
            metrics_df = pd.DataFrame([r['metrics'] for r in results])
            metrics_df.insert(0, 'Strategy', [r['name'] for r in results])
            
            st.dataframe(metrics_df, use_container_width=True, height=300)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Returns Comparison")
                fig = px.bar(
                    metrics_df,
                    x='Strategy',
                    y='Total Return (%)',
                    color='Strategy',
                    color_discrete_sequence=[r['color'] for r in results],
                    text='Total Return (%)'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Risk-Adjusted Returns")
                fig = px.bar(
                    metrics_df,
                    x='Strategy',
                    y='Sharpe Ratio',
                    color='Strategy',
                    color_discrete_sequence=[r['color'] for r in results],
                    text='Sharpe Ratio'
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("### Strategy Configurations")
            
            for i, r in enumerate(results):
                c = r['config']
                
                # Special display for CCI30
                if r['name'] == 'CCI30 Index':
                    st.markdown(f"""
                    <div style='background: white; padding: 2rem; border: 3px solid {r['color']}; 
                                border-radius: 10px; margin-bottom: 1.5rem;'>
                        <h3 style='margin: 0 0 1.5rem 0;'>{r['name']} (Benchmark)</h3>
                        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;'>
                            <div><strong>Type:</strong><br>Buy & Hold</div>
                            <div><strong>Rebalancing:</strong><br>None</div>
                            <div><strong>Tokens:</strong><br>Top 30</div>
                            <div><strong>Description:</strong><br>Passive Index</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Regular strategy display
                    st.markdown(f"""
                    <div style='background: white; padding: 2rem; border: 3px solid {r['color']}; 
                                border-radius: 10px; margin-bottom: 1.5rem;'>
                        <h3 style='margin: 0 0 1.5rem 0;'>{r['name']}</h3>
                        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;'>
                            <div><strong>Method:</strong><br>{c['method'].replace('_', ' ').title()}</div>
                            <div><strong>Frequency:</strong><br>{c['rebalance_freq'].title()}</div>
                            <div><strong>Top N:</strong><br>{c['top_n']}</div>
                            <div><strong>Rebalances:</strong><br>{r['metrics']['Rebalances']}</div>
                            <div><strong>BTC Cap:</strong><br><span style='color: #1e40af; font-weight: 800;'>{c.get('btc_cap', 1)*100:.0f}%</span></div>
                            <div><strong>ETH Cap:</strong><br><span style='color: #1e40af; font-weight: 800;'>{c.get('eth_cap', 1)*100:.0f}%</span></div>
                            <div><strong>BTC+ETH:</strong><br><span style='color: #10b981; font-weight: 800;'>{c.get('btc_eth_combined', 1)*100:.0f}%</span></div>
                            <div><strong>Max Any:</strong><br><span style='color: #f59e0b; font-weight: 800;'>{c.get('max_any', 1)*100:.0f}%</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if c.get('method') == 'manual' and c.get('manual_weights'):
                    st.markdown("**Manual Weights Specified:**")
                    manual_df = pd.DataFrame([
                        {'Token': k, 'Weight (%)': f"{v:.2f}%"}
                        for k, v in sorted(c['manual_weights'].items(), key=lambda x: x[1], reverse=True)
                    ])
                    st.dataframe(manual_df, use_container_width=True, hide_index=True)
    
    else:
        # WELCOME
        st.markdown("## 👋 Welcome!")
        st.markdown("### Configure settings in sidebar and click RUN BACKTEST")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**📅 Data Range**\n\n{min_date.strftime('%Y-%m-%d')}\n\nto\n\n{max_date.strftime('%Y-%m-%d')}")
        with col2:
            st.info(f"**📊 Tokens**\n\n{len(all_tokens)} available")
        with col3:
            st.info(f"**📈 Data Points**\n\n{len(data['date'].unique())} weeks")
        
        st.markdown("---")
        
        st.markdown("""
        ### ✨ Features
        
        - **📊 Market Cap Weighting** - Weight by market capitalization
        - **⚖️ Equal Weighting** - Equal allocation across tokens
        - **📉 EWMA Weighting** - Exponentially weighted moving average with square root methodology
        - **✍️ Manual Weights** - Set custom allocations
        - **🔒 Advanced Capping** - BTC, ETH, combined, and universal caps
        - **📥 Export Reports** - Download CSV and TXT reports
        - **📈 Rich Visualizations** - Interactive charts and tables
        - **🎯 CCI30 Benchmark** - Compare against industry standard index
        """)

if __name__ == "__main__":
    main()
