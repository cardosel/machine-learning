import statsmodels.api as sm
pisa = pd.DataFrame({"year": range(1975, 1988), 
                         "lean": [2.9642, 2.9644, 2.9656, 2.9667, 2.9673, 2.9688, 2.9696, 
                                  2.9698, 2.9713, 2.9717, 2.9725, 2.9742, 2.9757]})

print(pisa)
plt.scatter(pisa["year"], pisa["lean"])

# FIT THE LINEAR MODEL
y = pisa.lean # target
X = pisa.year  # features
X = sm.add_constant(X)  # add a column of 1's as the constant term

# OLS -- Ordinary Least Squares Fit
linear = sm.OLS(y, X)
# fit model
linearfit = linear.fit()
print(linearfit.summary())

# PREDICT THE TRAINING DATA
yhat = linearfit.predict(X)
print(yhat)
residuals = yhat - y

# HISTOGRAM OF RESIDUALS
plt.hist(residuals, bins=5)

# SUM OF SQUARES
# sum the (predicted - observed) squared
SSE = np.sum((yhat-y.values)**2)
# Average y
ybar = np.mean(y.values)

# sum the (mean - predicted) squared
RSS = np.sum((ybar-yhat)**2)

# sum the (mean - observed) squared
TSS = np.sum((ybar-y.values)**2)

# R-SQUARED
R2 = RSS/TSS

# VARIANCE OF COEFFICIENTS
# Compute SSE
SSE = np.sum((y.values - yhat)**2)
# Compute variance in X
xvar = np.sum((pisa.year - pisa.year.mean())**2)
# Compute variance in b1 
s2b1 = (SSE / (y.shape[0] - 2)) / xvar

# T-DISTRIBUTION
from scipy.stats import t

# 100 values between -3 and 3
x = np.linspace(-3,3,100)

# Compute the pdf with 3 degrees of freedom
print(t.pdf(x=x, df=3))
# Pdf with 3 degrees of freedom
tdist3 = t.pdf(x=x, df=3)
# Pdf with 30 degrees of freedom
tdist30 = t.pdf(x=x, df=30)

# Plot pdfs
plt.plot(x, tdist3)
plt.plot(x, tdist30)

# STATISTICAL SIGNIFICANCE OF COEFFICIENTS
tstat = linearfit.params["year"] / np.sqrt(s2b1)

# P-VALUE
# At the 95% confidence interval for a two-sided t-test we must use a p-value of 0.975
pval = 0.975

# The degrees of freedom
df = pisa.shape[0] - 2

# The probability to test against
p = t.cdf(tstat, df=df)
beta1_test = p > pval

