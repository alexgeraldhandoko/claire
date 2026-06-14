# Claire Project Report

## 1. Dataset Collection

Initial searches for complete orderbook, or even Level 10 orderbook data
for stock markets without passing through a paywall proved to be difficult.
Some possible credible sources that had historical datasets and/or API, even
with a paywall, are:
- NASDAQ TotalView - NASDAQ's full depth-of-book feed.
- LOBSTER - An academic/professional limit order book data service.
- DataBento - Market-data vendor with historical and live APIs
- NYSE TAQ - NYSE's historical feeds that provide several related market-data
streams into one combined feed. It provides complete depth-of-book data for
NYSE-family venues.

Binance's datasets seem to be worth trying, given the free nature of the data 
and the credibility of Binance being the largest crypto trading exchange.
Binance provides free API to previous day's data that is provided daily, so
a live paper trading simulation could use Binance's API for a "one-day-behind-
trading-simulation" (the model looks like it's trading live but actually it's
one-day behind, like watching a YouTube livestream that ended one day ago.).
Furthermore, they provide free historical LOB data in both daily and monthly 
formats for BTCUSDT perpetual futures.

However, the Binance dataset did not include enough data to construct a Level 10 
LOB. It was a top-of-book ticker data, where the only features available were
best bid, best bid quantity, best ask, and best ask quantity. Thus, we had to
look for a different dataset.

## 2. Dataset Processing