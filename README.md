# Stock Price Prediction with Random Forest

This project implements a stock price prediction model using Random Forest and other technical indicators. The model predicts the direction of stock market prices based on historical data.

## Research Paper

This project is inspired by the research paper:

**Predicting the direction of stock market prices using random forest**  
Luckyson Khaidem, Snehanshu Saha, Sudeepa Roy Dey  
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
   cd stock-price-prediction-forest
   ```

2. **Install required libraries:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your data:** Place your CSV files in the `Data` directory. The files should include stock price information.

2. **Run the training script:**

   ```bash
   python train.py
   ```

   This will process the data, train the Random Forest model, and print the model's accuracy.

## Files

- `train.py`: Main script for training the model.
- `expSmoothed.py`: Contains functions for applying exponential smoothing and calculating target values.
- `decisionTree.py`: Implements the Decision Tree classifier.
- `randomForest.py`: Implements the Random Forest classifier.
- `files.py`: Contains file paths for data.
- `requirements.txt`: Lists the required Python libraries.
