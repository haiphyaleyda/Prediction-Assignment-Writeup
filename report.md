# Prediction Assignment report

Rohaiphy Aleyda Kana
24-03-2025

```R
library(caret)
library(rpart)
library(randomForest)
library(readr)
set.seed(2004)
```

## Load Data
This step reads the training and testing datasets from CSV files.
```R
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```

## Data Preprocessing
### Convert Class Labels to Factor
Since the target variable `classe` is categorical, we convert it into a factor.
```R
training$classe <- as.factor(training$classe)
testing$classe <- as.factor(testing$classe)
```

### Convert Character Columns to Numeric and Handle Missing Values
Many columns might be stored as characters but represent numerical values. This might cause problems down the line. So, We convert them and handle empty strings by replacing them with `NA`.
```R
training[] <- lapply(training, function(x) {
  if (is.character(x)) {
    x <- ifelse(x == "", NA, x)  # Convert empty strings to NA
    as.numeric(x)  # Convert to numeric
  } else {
    x  # Keep original type
  }
})

testing[] <- lapply(testing, function(x) {
  if (is.character(x)) {
    x <- ifelse(x == "", NA, x)
    as.numeric(x)
  } else {
    x
  }
})
```

### Remove Columns with High NA Proportion
We remove features where more than 95% of the values are NA since they are unlikely to contribute meaningful information.
```R
na_threshold <- 0.95
na_counts <- colSums(is.na(training)) / nrow(training)
training <- training[, na_counts < na_threshold]

testing <- testing[, intersect(names(training), names(testing)), drop = FALSE]
```

## Train-Test Split
We split the training dataset into training (70%) and validation (30%) sets to evaluate model performance before testing.
```R
trainIndex <- createDataPartition(training$classe, p = 0.7, list = FALSE)
trainData <- training[trainIndex, ]
validData  <- training[-trainIndex, ]
```

## Decision Tree Model
A simple decision tree model is trained using cross-validation (5-fold) to classify observations.
```R
tree_model <- train(classe ~ ., data = trainData, method = "rpart",
                    trControl = trainControl(method = "cv", number = 5))

tree_predictions <- predict(tree_model, newdata = validData)

# Evaluate performance using a confusion matrix
confusionMatrix(tree_predictions, validData$classe)
```
```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1673    0    0    0    0
         B    1 1139    0    0    0
         C    0    0    0    0    0
         D    0    0    0    0    0
         E    0    0 1026  964 1082

Overall Statistics

               Accuracy : 0.6617
                 95% CI : (0.6494, 0.6738)
    No Information Rate : 0.2845
    P-Value [Acc > NIR] : < 2.2e-16

                  Kappa : 0.5694

 Mcnemar's Test P-Value : NA

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9994   1.0000   0.0000   0.0000   1.0000
Specificity            1.0000   0.9998   1.0000   1.0000   0.5857
Pos Pred Value         1.0000   0.9991      NaN      NaN   0.3522
Neg Pred Value         0.9998   1.0000   0.8257   0.8362   1.0000
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2843   0.1935   0.0000   0.0000   0.1839
Detection Prevalence   0.2843   0.1937   0.0000   0.0000   0.5220
Balanced Accuracy      0.9997   0.9999   0.5000   0.5000   0.7928
```
The decision tree model perform poorly as indicated by the confusion matrix and statistics. With overall accuracy of 66%, the model cannot predict C and D class.

## Random Forest Model
A more complex random forest model is trained with cross-validation to improve accuracy.
```R
rf_model <- train(classe ~ ., data = trainData, method = "rf",
                  trControl = trainControl(method = "cv", number = 5))

rf_predictions <- predict(rf_model, newdata = validData)

# Evaluate performance using a confusion matrix
confusionMatrix(rf_predictions, validData$classe)
```
```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1674    0    0    0    0
         B    0 1139    1    0    0
         C    0    0 1025    0    0
         D    0    0    0  964    0
         E    0    0    0    0 1082

Overall Statistics

               Accuracy : 0.9998
                 95% CI : (0.9991, 1)
    No Information Rate : 0.2845
    P-Value [Acc > NIR] : < 2.2e-16

                  Kappa : 0.9998

 Mcnemar's Test P-Value : NA

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   1.0000   0.9990   1.0000   1.0000
Specificity            1.0000   0.9998   1.0000   1.0000   1.0000
Pos Pred Value         1.0000   0.9991   1.0000   1.0000   1.0000
Neg Pred Value         1.0000   1.0000   0.9998   1.0000   1.0000
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2845   0.1935   0.1742   0.1638   0.1839
Detection Prevalence   0.2845   0.1937   0.1742   0.1638   0.1839
Balanced Accuracy      1.0000   0.9999   0.9995   1.0000   1.0000
```
The random forest model clearly performs better than the DT model. Thus, this model is used for prediction of the testing data.

## Predictions on Testing Data
We use the trained random forest model to predict class labels for the testing dataset and save the results.
```R
predictions <- predict(rf_model, newdata = testing)
testing$Predicted_Classe <- predictions
write.csv(testing, "predictions.csv", row.names = FALSE)
```
