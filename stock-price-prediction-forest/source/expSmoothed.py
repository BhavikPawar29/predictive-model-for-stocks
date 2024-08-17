import pandas as pd
import numpy as np
from techIndicators import calculate_rsi, calculate_stochastic_oscillator, calculate_williams_r, calculate_macd, calculate_proc, calculate_obv

#Data Preprocessing

def exponential_smoothing(series, alpha):
    """
    Exponential smoothing applies more weightage to the recent 
    observation and exponentially decreasing weights to past observations.
    (from 3.1 in paper)

    """
    smoothed = series.copy()
    for t in range(1, len(series)):
        smoothed.iloc[t] = alpha * series.iloc[t] + (1 - alpha) * smoothed.iloc[t - 1]
    return smoothed


def apply_smoothing_and_targeting(file_path, alpha, d):
    """
    d is the number of days after which the prediction is to be made. When the value of targeti is
    +1, it indicates that there is a positive shift in the price after d days and -1 indicates that there is
    a negative shift after d days.
    (from 3.1 in paper)

    """
    # Load the data into a pandas DataFrame
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Apply exponential smoothing to the 'Close' price
    df['Smoothed_Close'] = exponential_smoothing(df['Close'], alpha)
    df['Smoothed_PrevClose'] = exponential_smoothing(df['Prev Close'], alpha)


    # Technical indicator example: #Feature Extraction
    # Calculate technical indicators

    df['RSI'] = calculate_rsi(df['Close'])
    df['%K'] = calculate_stochastic_oscillator(df)
    df['%R'] = calculate_williams_r(df)
    df[['MACD', 'SignalLine']] = calculate_macd(df)
    df['PROC'] = calculate_proc(df['Close'], 14)
    df['OBV'] = calculate_obv(df)

    # Calculate target values
    df['target'] = df['Close'].shift(-d) - df['Close']
    df['target'] = np.sign(df['target'])

     # Handle NaN or infinite values
    df['target'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['target'].fillna(0, inplace=True)  # Assuming 0 as the neutral value; adjust as needed
    #df['target'] = df['target'].astype(int)  # Ensure labels are integers

    # Map -1 to 0
    df['target'] = df['target'].replace(-1, 0).astype(int)

    # Drop NaN values resulting from shifting
    df.dropna(inplace=True)

    # Prepare feature matrix and labels
    features = df[['Smoothed_Close', 'Smoothed_PrevClose', 'RSI', '%K', '%R', 'MACD', 'SignalLine', 'PROC', 'OBV']]
    labels = df['target']

    return features, labels

def process_all_files(file_paths, alpha, d, n):
    """
    Processes the first 'n' files from the list of file paths.
    
    :param file_paths: List of file paths to process
    :param alpha: Smoothing factor for exponential smoothing
    :param d: Number of days after which the prediction is to be made
    :param n: Number of files to process from the beginning of the list
    :return: Numpy arrays containing features and labels combined from all processed files
    """
    # Slice the file_paths list to get the first 'n' files
    file_paths_to_process = file_paths[:n]

    all_features = []
    all_labels = []

    for file_path in file_paths_to_process:
        features, labels = apply_smoothing_and_targeting(file_path, alpha, d)
        
        # Ensure the features and labels are in DataFrame format before concatenation
        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(labels, pd.Series):
            labels = labels.values
        
        all_features.append(features)
        all_labels.append(labels)
        print(f'Processed {file_path}')

    # Concatenate features and labels into numpy arrays
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)

    return combined_features, combined_labels
