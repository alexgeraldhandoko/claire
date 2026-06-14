# Claire Project Report

## 0. Project Direction

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

Subsequent searches revealed that reliable datasets for crypto assets are more 
publicly available and do not usually sit behind a paywall. Thus, a decision
was made to construct the model to predict fair-value movement of crypto assets
instead of stocks. One such crypto asset would be BTCUSDT perpetual futures.

BTCUSDT perpetual futures were chosen over BTCUSDT spot market because perpetual
futures markets generally have higher trading activity, deeper liquidity, and
more frequent order book updates. This makes them more suitable for a limit order
book prediction model as the model is expected to learn better from dense market 
microstructure data, such as rapidly changing bid and ask prices, quantities, 
spreads, and order book imbalance.

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

Next, we would like to try the Bybit dataset. Bybit is the second largest crypto
exchange and also provides free historical data and live orderbook data API. 
The Bybit data proved to be more comprehensive than the Binance dataset. It provided
complete depth-of-book data, which is more than sufficient for this project.

Thus, the Bybit historical LOB dataset for BTCUSDT perpetual futures was chosen 
as the source of data for this project. 

## 2. Dataset Processing

**Dataset Parsing**
The LOB dataset came in a huge JSON file, where each line represents the LOB
or a change to the LOB at every 0.1s interval of the day.

Since the dataset was large, parsing it initially took a long time. It was
thus optimised over several iterations. The optimisations included:
- Using orjson library instead of json library for highly optimised parsing
- Using Numpy methods instead of manual iterative appending
- Removing use of Pandas
- Reduced print statements

**Label Engineering**
We have decided to use three classification labels for the input:
DOWN/STATIONARY/UP

The reason is because the model minimally needs to be able to classify the
direction of the fair-value price movement reliably enough before it can 
predict the direction together with the amount of movement. This is because 
if the model is not able to predict the direction of the movement reliably, 
it will not be able to predict the actual movement either to the extent that
it can be used as a factor in making profitable trading decisions.

There were several methods available to label the classification ground truth 
for the input data:
1. Absolute threshold

**Feature Scaling**
Feature scaling remains important although many of the features are price levels 
hovering around the same magnitude. This is because the features contain a mix
of both prices and also share amounts.

Feature Transformation

Processed Dataset Storage