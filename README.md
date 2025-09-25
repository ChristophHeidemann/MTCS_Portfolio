# Monte Carlo Portfolio Simulation

This repository contains a **Monte Carlo simulation** to model the future value of a stock portfolio based on historical stock data. The simulation uses historical returns and covariance between stocks to project potential portfolio outcomes over time.

---

## Features

- Fetch historical stock price data from Yahoo Finance.
- Calculate daily returns, mean returns, and covariance matrix.
- Simulate thousands of potential portfolio paths using Monte Carlo methods.
- Visualize projected portfolio values over a specified timeframe.
- Supports custom stock lists, timeframes, and initial investment amounts.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ChristophHeidemann/MTCS_Portfolio
cd monte-carlo-portfolio
```
2. Install dependencies (Python 3.9+ recommended):

```bash
pip install pandas numpy matplotlib yfinance
````

## How it works

1. Data Collection:
   
   Historical stock closing prices are fetched using yfinance. Daily returns, mean returns, and covariance matrices are calculated.
3. Monte Carlo Simulation:
   - Random portfolio weights are generated and normalized.
   - Daily returns are simulated using a multivariate normal distribution via Cholesky decomposition.
   - Cumulative portfolio values are calculated for each simulation.
4. Visualization:
   
   A line plot shows thousands of simulated portfolio paths, representing potential future outcomes.
