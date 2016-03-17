# MULTICLASS CLASSIFICATION
import pandas
import numpy as np

# Filename
auto_file = "auto.txt"

# Column names, not included in file
names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
         'year', 'origin', 'car_name']

# Read in file
# Delimited by an arbitrary number of whitespaces
auto = pandas.read_table(auto_file, delim_whitespace=True, names=names)

# Show the first 5 rows of the dataset
print(auto.head())

unique_regions = auto["origin"].unique()

# CLEAN DATA SET
# Delete the column car_name
del auto["car_name"]
# Remove rows with missing data
auto = auto[auto["horsepower"] != '?']

# USING DUMMY VARIABLES
# input a column with categorical variables
def create_dummies(var):
    # get the unique values in var and sort
    var_unique = var.unique()
    var_unique.sort()

    # Initialize a dummy DataFrame to store variables
    dummy = pandas.DataFrame()

    # loop through all but the last value
    for val in var_unique[:-1]:
        # which columns are equal to our unique value
        d = var == val

        # make a new column with a dummy variable
        dummy[var.name + "_" + str(val)] = d.astype(int)

    # return dataframe with our dummy variables
    return(dummy)

# lets make a copy of our auto dataframe to modify with dummy variables
modified_auto = auto.copy()

# make dummy varibles from the cylinder categories
cylinder_dummies = create_dummies(modified_auto["cylinders"])

# merge dummy varibles to our dataframe
modified_auto = pandas.concat([modified_auto, cylinder_dummies], axis=1)

# delete cylinders column as we have now explained it with dummy variables
del modified_auto["cylinders"]

print(modified_auto.head())
# make dummy varibles from the cylinder categories
year_dummies = create_dummies(modified_auto["year"])

# merge dummy varibles to our dataframe
modified_auto = pandas.concat([modified_auto, year_dummies], axis=1)

# delete cylinders column as we have now explained it with dummy variables
del modified_auto["year"]

# MULTICLASS CLASSIFICATION
# get all columns which will be used as features, remove 'origin'
features = np.delete(modified_auto.columns, modified_auto.columns == 'origin')

# shuffle data
shuffled_rows = np.random.permutation(modified_auto.index)

# Select 70% of the dataset to be training data
highest_train_row = int(modified_auto.shape[0] * .70)
# Select 70% of the dataset to be training data
train = modified_auto.loc[shuffled_rows[:highest_train_row], :]

# Select 30% of the dataset to be test data
test = modified_auto.loc[shuffled_rows[highest_train_row:], :]

# TRAINING A MULTICLASS LOGISTIC REGRESSION MODEL
from sklearn.linear_model import LogisticRegression

# find the unique origins
unique_origins = modified_auto["origin"].unique()
unique_origins.sort()

# dictionary to store models
models = {}

for origin in unique_origins:
    # initialize model to dictionary
    models[origin] = LogisticRegression()

    # select columns for predictors and predictands
    X_train = train[features]
    y_train = train["origin"] == origin

    # fit model with training data
    models[origin].fit(X_train, y_train)

# Dataframe to collect testing probabilities
testing_probs = pandas.DataFrame(columns=unique_origins)
for origin in unique_origins:

    # select testing features
    X_test = test[features]

    # compute probability of observation being in the origin
    testing_probs[origin] = models[origin].predict_proba(X_test)[:,1]

# CHOOSE THE ORIGIN
predicted_origins = testing_probs.idxmax(axis=1)

# CONFUSION MATRIX
# Remove pandas indicies
predicted_origins = predicted_origins.values
origins_observed = test['origin'].values

# fill in this confusion matrix
confusion = pandas.DataFrame(np.zeros(shape=(unique_origins.shape[0], unique_origins.shape[0])),
                             index=unique_origins, columns=unique_origins)
# Each unique prediction
for pred in unique_origins:
    # Each unique observation
    for obs in unique_origins:
        # Check if pred was predicted
        t_pred = predicted_origins == pred
        # Check if obs was observed
        t_obs = origins_observed == obs
        # True if both pred and obs
        t = (t_pred & t_obs)
        # Count of the number of observations with pred and obs
        confusion.loc[pred, obs] = sum(t)

# false positives = observed 1 and predicted 2 or 3
fp1 = confusion.ix[2,[1,3]].sum()

# AVERAGE ACCURACY
# The total number of observations in the test set
n = test.shape[0]
# Variable to store true predictions
sumacc = 0
# Loop over each origin
for i in confusion.index:
    # True Positives
    tp = confusion.loc[i, i]
    # True negatives
    tn = confusion.loc[unique_origins[unique_origins != i], unique_origins[unique_origins != i]]
    # Add the sums
    sumacc += tp.sum() + tn.sum().sum()

# Compute average accuracy
avgacc = sumacc/(n*unique_origins.shape[0])
print(avgacc)

# PRECISION AND RECALL
# Variable to add all precisions
ps = 0
# Loop through each origin (class)
for j in confusion.index:
    # True positives
    tps = confusion.ix[j,j]
    # Positively predicted for that origin
    positives = confusion.ix[j,:].sum()
    # Add to precision
    ps += tps/positives

# divide ps by the number of classes to get precision
precision = ps/confusion.shape[0]
print('Precision = {0}'.format(precision))
# Variable to add all recalls
rcs = 0
for j in confusion.index:
    # Current number of true positives
    tps = confusion.ix[j,j]
    # True positives and false negatives
    origin_count = confusion.ix[:,j].sum()
    # Add recall
    rcs += tps/origin_count

# Compute recall
recall = rcs/confusion.shape[0]

# F-SCORE
# Variable to add all precisions
scores = []
# Loop through each origin (class)
for j in confusion.index:
    # True positives
    tps = confusion.ix[j,j]
    # Positively predicted for that origin
    positives = confusion.ix[j,:].sum()
    # True positives and false negatives
    origin_count = confusion.ix[:,j].sum()
    # Compute precision
    precision = tps / positives
    # Compute recall
    recall = tps / origin_count
    # Append F_i score
    fi = 2*precision*recall / (precision + recall)
    scores.append(fi)
shape = modified_auto.shape


# Average over all scores
fscore = np.mean(scores)

# METRICS
# for all those metrics above, you can just use sklearn's built-in
# packages to find the metric you're looking for
from sklearn.metrics import precision_score, recall_score, f1_score

# Compute precision score with micro averaging
pr_micro = precision_score(test["origin"], predicted_origins, average='micro')
pr_weighted = precision_score(test["origin"], predicted_origins, average='weighted')
rc_weighted = recall_score(test["origin"], predicted_origins, average='weighted')
f_weighted = f1_score(test["origin"], predicted_origins, average='weighted')
