import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

# import data
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)
    stockData = stockData['Close'] # type: ignore
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov() # type: ignore
    return meanReturns, covMatrix

stockList = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365*3)

meanReturns, covMatrix = get_data(stockList, startDate, endDate)

#print("Mean Returns:\n", meanReturns)
#print("\nCovariance Matrix:\n", covMatrix)

weights = np.random.random(len(stockList))
weights /= np.sum(weights)

#print("\nRandom Weights:\n", weights)

# monte carlo simulation
#number of simulations
mc_sims = 10000
T = 365 # timefrae in  days

meanM = np.full(shape=(T, len(stockList)), fill_value=meanReturns) # type: ignore
meanM = meanM.T # type: ignore

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPortfolio = 10000

for m in range(mc_sims):
    #mc loops
    Z = np.random.normal(size=(T, len(stockList)))
    L = np.linalg.cholesky(covMatrix) # type: ignore
    dailyReturns = meanM + np.inner(L, Z) # type: ignore
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1)*initialPortfolio # type: ignore

plt.plot(portfolio_sims)
plt.xlabel('Days') 
plt.ylabel('Portfolio Value ($)')
plt.title('Monte Carlo Simulation of Portfolio Value Over Time')
plt.show()