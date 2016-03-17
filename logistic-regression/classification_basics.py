# Import matplotlib for plotting
import matplotlib.pyplot as plt

# We will use pandas to work with the data
import pandas

# Read file
credit = pandas.read_csv("credit.csv")

# Dataframe of our data
# credit["model_score"] is the probability provided by the model
# credit["paid"] is the observed payments
# .head(10) shows the first 10 rows of the dataframe
print(credit.head(10))

plt.scatter(credit["model_score"], credit["paid"])
plt.show()


# PREDICTIVE POWER
# Will we approve the credit card based on the probability of paying?
pred = credit["model_score"] > 0.5

# This operation tells us whether the prediction was correct
print(pred == credit["paid"])
accuracy = sum(pred == credit["paid"]) / len(pred)

# BINARY DISCRIMINATION
# prediction with discrimination threshold at 0.50
pred = credit["model_score"] > 0.5

# number of true positives
TP = sum(((pred == 1) & (credit["paid"] == 1)))
print(TP)

FN = sum(((pred == 0) & (credit["paid"] == 1)))

# SENSITIVITY, SPECIFICITY, AND FALL-OUT
# Predicted to play tennis
pred = credit["model_score"] > 0.5

# Number of true negatives
TN = sum(((pred == 0) & (credit["paid"] == 0)))

# Number of false positives
FP = sum(((pred == 1) & (credit["paid"] == 0)))
FN = sum(((pred == 0) & (credit["paid"] == 1)))
# Compute the false positive rate
FPR = FP / (TN + FP)
print(FPR)

TP = sum(((pred == 1) & (credit["paid"] == 1)))
TPR = TP/ (TP + FN)

# ROC CURVES
def roc_curve(observed, probs):
    # choose thresholds between 0 and 1 to discriminate prediction
    thresholds = numpy.asarray([(100-j)/100 for j in range(100)])

    # initialize false and true positive rates
    fprs = numpy.asarray([0. for j in range(100)])
    tprs = numpy.asarray([0. for j in range(100)])

    # Loop through each threshold
    for j, thres in enumerate(thresholds):
        # Using the new threshold compute predictions
        pred = probs > thres
        # Count the Number of False Positives
        FP = sum((observed == 0) & (pred == 1))
        # Count the Number of True Negatives
        TN = sum((observed == 0) & (pred == 0))
        # Compute the False Postive Rate
        FPR =  float(FP / (FP + TN))
        # Count the number of True Positives
        TP = sum((observed == 1) & (pred == 1))
        # Count the number of False Negatives
        FN = sum((observed == 1) & (pred == 0))
        # Compute the True Positive Rate
        TPR = float(TP / (TP + FN))
        # Store the true and false positive rates
        fprs[j] = FPR
        tprs[j] = TPR

    return fprs, tprs, thresholds

fpr, tpr, thres = roc_curve(credit["paid"], credit["model_score"])
idx = numpy.where(fpr>0.20)[0][0]
print(tpr[idx])
print(thres[idx])
plt.plot(fpr, tpr)

# AREA UNDER THE CURVE
from sklearn.metrics import roc_auc_score

probs = [ 0.98200848,  0.92088976,  0.13125231,  0.0130085,   0.35719083,
         0.34381803, 0.46938187,  0.53918899,  0.63485958,  0.56959959]
obs = [1, 1, 0, 0, 1, 0, 1, 0, 0, 1]

testing_auc = roc_auc_score(obs, probs)
print("Example AUC: {auc}".format(auc=testing_auc))

auc = roc_auc_score(credit["paid"], credit["model_score"])

# PRECISION AND RECALL
pred = credit["model_score"] > 0.5

# True Positives
TP = sum(((pred == 1) & (credit["paid"] == 1)))
print(TP)

# False Positives
FP = sum(((pred == 0) & (credit["paid"] == 1)))
print(FP)

# False Negatives
FN = sum(((pred == 1) & (credit["paid"] == 0)))
print(FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)

# PRECISION AND RECALL CURVE
from sklearn.metrics import precision_recall_curve

probs = [ 0.98200848,  0.92088976,  0.13125231,  0.0130085,   0.35719083,
         0.34381803, 0.46938187,  0.53918899,  0.63485958,  0.56959959]
obs = [1, 1, 0, 0, 1, 0, 1, 0, 0, 1]

precision, recall, thresholds = precision_recall_curve(obs, probs)
plt.plot(recall, precision)
plt.show()
precision, recall, thresholds = precision_recall_curve(credit["paid"], credit["model_score"])
plt.plot(recall, precision)

# ADMISSIONS ROC CURVE
from sklearn.metrics import roc_curve, roc_auc_score

# Compute the probabilities predicted by the training and test set
# predict_proba returns probabilies for each class.  We want the second column
train_probs = logistic_model.predict_proba(data_train[['gpa', 'gre']])[:,1]
test_probs = logistic_model.predict_proba(data_test[['gpa', 'gre']])[:,1]
# Compute auc for training set
auc_train = roc_auc_score(data_train["admit"], train_probs)

# Compute auc for test set
auc_test = roc_auc_score(data_test["admit"], test_probs)

# Difference in auc values
auc_diff = auc_train - auc_test

# Compute ROC Curves
roc_train = roc_curve(data_train["admit"], train_probs)
roc_test = roc_curve(data_test["admit"], test_probs)

# Plot false positives by true positives
plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_test[0], roc_test[1])
