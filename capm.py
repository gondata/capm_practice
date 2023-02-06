# First of all we have to import the libraries that we are going to use

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()

# Then we have to download the data

tickers = ['AAPL', '^GSPC']
startdate = '2012-01-01'
enddate = '2023-02-02'
data = pd.DataFrame()

for t in tickers:
    data[t] = pdr.get_data_yahoo(t, start=startdate, end=enddate)['Adj Close']

# Values

log_returns = np.log(1+data.pct_change())
cov = log_returns.cov()*252

# Model

cov_market = cov.iloc[0, 1]
market_var = log_returns['^GSPC'].var()*252
stock_beta = cov_market/market_var
rf = 0.0352       # Risk free rate
risk_premium = (log_returns['^GSPC'].mean()*252) - rf


stock_capm_return = rf +stock_beta * risk_premium
sharpe = (stock_capm_return - rf) / (log_returns['AAPL'].std()*252**0.5)

print("The Beta of " + str(tickers) + "is " + str(round(stock_beta, 3)))
print("The CAPM Return of " + str(tickers) + "is " + str(round(stock_capm_return*100, 3)) + "%")
print("The Sharpe Ratio of " + str(tickers) + "is " + str(round(sharpe, 3)))