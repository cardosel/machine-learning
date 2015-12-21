# code from Machine Learning in R

# import data
insurance <- read.csv("insurance.csv", stringsAsFactors = TRUE)
str(insurance)

# summarize the charges variable
summary(insurance$charges)

# histogram of insurance charges
hist(insurance$charges)

# table of region
table(insurance$region)

# exploring relationships among features: correlation matrix
cor(insurance[c("age", "bmi", "children", "charges")])

# visualing relationships among features: scatterplot matrix
pairs(insurance[c("age", "bmi", "children", "charges")])

# more informative scatterplot matrix
install.packages("psych")
library(psych)
pairs.panels(insurance[c("age", "bmi", "children", "charges")])

## Step 3: Training a model on the data ----
ins_model <- lm(charges ~ age + children + bmi + sex + smoker + region,
                data = insurance)
ins_model <- lm(charges ~ ., data = insurance) # this is equivalent to above

# see the estimated beta coefficients
ins_model

## Step 4: Evaluating model performance ----
# see more detail about the estimated beta coefficients
summary(ins_model)

## Step 5: Improving model performance ----

# add a higher-order "age" term
insurance$age2 <- insurance$age^2

# add an indicator for BMI >= 30
insurance$bmi30 <- ifelse(insurance$bmi >= 30, 1, 0)

# create final model
# remember bmi30*smoker includes their interaction along with each individually
ins_model2 <- lm(charges ~ age + age2 + children + bmi + sex +
                   bmi30*smoker + region, data = insurance)

summary(ins_model2)