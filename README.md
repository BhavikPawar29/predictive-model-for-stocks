# Stock Price Prediction with Random Forest

## Overview
This project applies a Random Forest model to predict stock market trends using various technical indicators. The dataset used includes historical stock price data, and the implementation features a custom Random Forest classifier along with essential technical indicators for feature extraction.

## Dataset

### About Dataset
**Context**  
Stock market data is widely analyzed for educational, *business, and personal interests.

**Content**  
The dataset includes price history and trading volumes for fifty stocks in the NIFTY 50 index from the NSE (National Stock Exchange) of India. It contains day-level data with pricing and trading values split across CSV files for each stock, and a metadata file with macro-information about the stocks. The data spans from January 1, 2000, to April 30, 2021.

**Update Frequency**  
The dataset is updated once a month to ensure the latest and most relevant information.

**Acknowledgements**  
NSE India: [https://www.nseindia.com/](https://www.nseindia.com/)

## Research Paper

This project is inspired by the research paper:

**Predicting the direction of stock market prices using Random Forest**  
Luckyson Khaidem, Snehanshu Saha, Sudeepa Roy Dey [2016]
[khaidem90@gmail.com](mailto:khaidem90@gmail.com), [snehanshusaha@pes.edu](mailto:snehanshusaha@pes.edu), [sudeepar@pes.edu](mailto:sudeepar@pes.edu)

## Features

- Exponential smoothing of historical stock prices
- Calculation of technical indicators such as RSI, MACD, Stochastic Oscillator, Williams %R, PROC, and OBV
- Random Forest model for predicting stock price direction
- Out-of-Bag (OOB) error calculation for model evaluation

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/BhavikPawar29/predictive-model-for-stocks.git
   ```

2. **Install required libraries:**

   The `requirements.txt` file is located in the root directory of the repository, outside the `source` folder. Install the required libraries with:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your data:** Place your CSV files in the `Data` directory. The files include stock price information.

2. Edit the `files.py` file to include the paths to your data files. You can specify your file paths in the file_paths variable.

3.  **Run the training script:**

   ```bash
   cd predictive-model-for-stocks/source
   python train.py
   ```

   This will process the data, train the Random Forest model, and print the model's accuracy and OOB error.

## Files

- `train.py`: Main script for training the model.
- `expSmoothed.py`: Contains functions for applying exponential smoothing and calculating target values.
- `decisionTree.py`: Implements the Decision Tree classifier.
- `randomForest.py`: Implements the Random Forest classifier.
- `files.py`: Contains file paths for data.
- `requirements.txt`: Lists the required Python libraries (located in the root directory).
- `techIndicators.py`: Includes implementations for various technical indicators used in the model.
