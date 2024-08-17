import numpy as np

def calculate_rsi(series, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given price series.

    RSI is a momentum oscillator that measures the speed and change of price movements.
    It ranges from 0 to 100 and is used to identify overbought or oversold conditions in a market.

    Parameters:
    - series (pandas.Series): A pandas Series containing the price data (typically closing prices).
    - window (int, optional): The number of periods to use for calculating the RSI. Default is 14.

    Returns:
    - pandas.Series: A Series containing the RSI values. The RSI values are computed over the specified window period.
    
    Notes:
    - The RSI is calculated using the average gain and loss over the specified window period.
    - The result is a Series with the same length as the input series, where each value represents the RSI at that point in time.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_stochastic_oscillator(df, window=14):
    """
    This indicator measures the level of the closing 
    price relative to the price range over a specified period.

    Parameters:
    - df: DataFrame containing the stock data.
    - window: The period over which to calculate the low and high range (default is 14 days).

    Returns:
    - %K: The Stochastic Oscillator value.
    """
    df['L14'] = df['Low'].rolling(window=window, min_periods=1).min()
    df['H14'] = df['High'].rolling(window=window, min_periods=1).max()
    df['%K'] = 100 * (df['Close'] - df['L14']) / (df['H14'] - df['L14'])
    return df['%K']

def calculate_williams_r(df, window=14):
    """
    Williams %R is a momentum indicator
    that measures overbought and oversold levels.

    Parameters:
    - df: DataFrame containing the stock data.
    - window: The period over which to calculate the low and high range (default is 14 days).

    Returns:
    - %R: The Williams %R value.
    """
    df['L14'] = df['Low'].rolling(window=window, min_periods=1).min()
    df['H14'] = df['High'].rolling(window=window, min_periods=1).max()
    df['%R'] = -100 * (df['H14'] - df['Close']) / (df['H14'] - df['L14'])
    return df['%R']

def calculate_macd(df, ema_short_period=12, ema_long_period=26, signal_period=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) and Signal Line.

    Parameters:
    - df: DataFrame containing the stock data.
    - ema_short_period: The period for the short-term EMA (default is 12 days).
    - ema_long_period: The period for the long-term EMA (default is 26 days).
    - signal_period: The period for the Signal Line EMA (default is 9 days).

    Returns:
    - df: DataFrame with MACD and Signal Line columns added.
    """
    # Calculate short-term and long-term EMAs
    df[f'EMA{ema_short_period}'] = df['Close'].ewm(span=ema_short_period, adjust=False).mean()
    df[f'EMA{ema_long_period}'] = df['Close'].ewm(span=ema_long_period, adjust=False).mean()
    
    # Calculate MACD line
    df['MACD'] = df[f'EMA{ema_short_period}'] - df[f'EMA{ema_long_period}']
    
    # Calculate Signal Line
    df['SignalLine'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    
    return df[['MACD', 'SignalLine']]

def calculate_proc(series, n):
    """
    Calculate the Price Rate of Change (PROC) for a given series.

    Parameters:
    - series: The series of closing prices.
    - n: The number of days over which the price change is measured.

    Returns:
    - PROC: The Price Rate of Change.
    """
    proc = (series - series.shift(n)) / series.shift(n) * 100
    return proc

def calculate_obv(df):
    """
    Calculate the On Balance Volume (OBV) for a given DataFrame.

    Parameters:
    - df: DataFrame containing the stock data.

    Returns:
    - OBV: The On Balance Volume value.
    """
    df['OBV'] = (df['Volume'] * np.where(df['Close'] > df['Close'].shift(1), 1,
                                          np.where(df['Close'] < df['Close'].shift(1), -1, 0))).cumsum()
    return df['OBV']
