import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime


# Get data from Quandl and generate some more valuable data from it.
df = quandl.get("WIKI/GOOGL")
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# Strip out the data we don't want.
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# Set up our "label" column.
# Label is ML speak for the output.
# In contrast to features, which are the input.
# X => features
# Y => labels
forecast_col = 'Adj. Close'
# We use -99999 so that it will be removed as an outlier by most ML programs
# We can't just pass NaN
df.fillna(value=-999999, inplace=True)
# The length of our forecast is 1% the length of our input
forecast_out = int(math.ceil(0.01 * len(df)))
# Create the output rows
df['label'] = df[forecast_col].shift(-forecast_out)

# Scale the input and output features in the same call, then split them
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Drop any remaining NaN
df.dropna(inplace=True)

y = np.array(df['label'])

# Split into training and test sets, run classifier, check confidence
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

print("Confidence: ", confidence)

# Predict price into the future
forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

# Fixup data frame for plotting
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show("plot.jpg")
