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
1. Absolute change compared to next timestep's mid price
e.g. Mid price change of > 0.001 as up, < -0.001 as down, and stationary
otherwise.
This may work poorly for different datasets. For instance, 0.001 is a very small
change for assets with high prices like bitcoin
2. Percentage change compared to next timestep's mid price
e.g. Mid price change of > 0.2% as up, < -0.2% as down, and stationary
otherwise.
This may result in noisy labelling. Financial data is highly stochastic, and 
random price fluctuations can happen from one timestep to the next. Thus,
a +0.2% change could be caused by noise instead of real market factors.
3. Percentage change compared to the smoothed future average mid price
e.g. Compare the current mid price with the average price of the next 50 
timesteps. If the percentage change is > 0.2%, then label as up. If the 
percentage change is < -0.2%, label as down. Otherwise, label as stationary.
This labelling method could be fine for academic research. But there has been
criticisms regarding this method when used to build models for the profitable 
trading. One of them is that this method ignores transaction costs. For
instance, if the spread relative to mid price is 0.5%, then the costs could
outweigh the earnings from a 0.2% movement in mid price.
4. Difference in the percentage change of current mid price and the average spread
as a percentage of the mid price across the dataset
e.g. Average spread as percentage of mid price is calculated to be 0.5% 
for the entire dataset. If the mid price percentage change from one timestep to the
next timestep is > 0.5%, label as up. If < -0.5%, label as down. Otherwise,
label as stationary.
This takes into account possible transaction costs of the trade.

The 4th option will be used for this project due to its comprehensiveness as
compared to earlier options.

**Feature Scaling**
Feature scaling remains important although many of the features are price levels 
hovering around the same magnitude. This is because the features contain a mix
of both prices and also share amounts that can differ in magnitudes.

This project will use feature scaling as provided by Sklearn's StandardScaler,
which does standardisation (z-score scaling) of the features.

**Feature Engineering**
The order book provides the bulk of the features. However, there are still
several handcrafted feature transformations and additional features that 
should/could be used by the model:

1. Replace absolute prices with relative price distances
The target used during learning depends on whether the future mid price
moved enough relative to the current mid price such that it can overcome
transaction costs caused by the spread. This requires the model to analyse
and interpret factors from the data such as:
- How close is bid/ask side liquidity from the mid price?
- How thin/thick is the bid/ask side?
Thus, using ask/bid levels relative to the current mid price instead of
absolute ask/bid levels may be useful to represent the structure of the 
order book to the model.

Moreover, one LOB paper by Avraam Tsantekidis proposed that stationary features 
could be useful for LOB prediction since financial data is non-stationary. 
This means that although BTCUSDT prices may fluctuate vastly across time, the 
relative bids and asks around the mid price as percentages remain more stable. 
This means the model can be applied to more contexts across time regardless of the 
price of BTC.

However, it is not sufficiently rigorous to claim that the model could predict
better when passed in the relative price distances as opposed to the actual
absolute prices. In fact, this point applies to all features that are proposed to 
be added. This is because the model may learn better transformations
with less engineered features that help it to make better predictions. Thus,
this project will test the model validation performance on a variety of feature
formats.

2. Add spread as percentage of current mid price.
The classification labels predict UP/DOWN/STATIONARY based on whether the mid price
moved enough to overcome the transaction costs represented by the spread. Thus, it
is useful for the model to know the current spread in order to predict whether
the mid price in the next time step will move enough to overcome the spread.

3. Add depth imbalance of bid and ask side
Comparing the bid vs ask side provides a probabilistic clue to the future movement
direction of the asset's fair value. This is because a larger bid size means there
is more liquidity to absorb downward pressure from the ask side in cases of incoming
aggressive sell orders and vice versa.

To measure the relative sizes of the bid and ask sides, we will use depth imbalance.
It is calculated as follows:

Total bid depth = sum of all bid sizes
Total ask depth = sum of all ask sizes
Depth imbalance = (
   (total bid depth - total ask depth) /
   (total bid depth + total ask depth)
)

This gives the model a measure of the relative thickness of the bid versus ask sides.
This feature is chosen because it gives a nice value for the model, as it is:
- Centered around zero
- Bounded between -1 and 1
- Symmetric. This means that if ask depth is x times larger than bid depth, the
magnitude of depth imbalance is the same as if the bid depth is x times larger 
than ask depth.

We would also add depth imbalance for different depths of the order book so that
the model can see summaries of the bid vs ask pressure. This could provide important
information for the model's prediction task as the imabalances across different levels 
of the book may display different signals, e.g.:
Let depth_imbalance_x be the depth imbalance at levels 1 to ```x``` of the order book.
- If depth_imbalance_1 is high, then the market looks very bullish around the current
price
- However, if depth_imbalance_10 is very low, then the wider book may actually resist 
the upward pressure from the ask side.

**Processed Dataset Storage**
Since the project will involve testing different architectures on training data,
the preprocessing step needs to be repeatable and consistent. This means that we 
will store the preprocessing pipelines in a separate file to maintain modularity
using joblib.dump

Several processing pipelines are also available so that a single architecture can
be tested on several different pipelines. Each pipeline will consist of different
feature transformations.

The project result would therefore be displayed in a performance matrix, where the
rows are the different model architectures, the columns represent the processing
pipeline used to process the raw dataset, and each cell being the validation
performance of the model.

The processed dataset itself will also be stored in separate files as PyTorch
tensors and can be loaded using torch.load()

## 3. Model building