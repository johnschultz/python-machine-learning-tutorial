import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

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

print(df.tail())

# Drop any remaining NaNs
df.dropna(inplace=True)

# Create out input and output numpy arrays (needed to pass to sklearn)
X = np.array(df.drop(['label'], 1))
# It is customary to scale inputs to between -1 and 1
X = preprocessing.scale(X)
y = np.array(df['label'])


# Create the training and test sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)
