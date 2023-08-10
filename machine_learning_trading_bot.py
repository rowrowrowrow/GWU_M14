#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Trading Bot
# 
# In this Challenge, you’ll assume the role of a financial advisor at one of the top five financial advisory firms in the world. Your firm constantly competes with the other major firms to manage and automatically trade assets in a highly dynamic environment. In recent years, your firm has heavily profited by using computer algorithms that can buy and sell faster than human traders.
# 
# The speed of these transactions gave your firm a competitive advantage early on. But, people still need to specifically program these systems, which limits their ability to adapt to new data. You’re thus planning to improve the existing algorithmic trading systems and maintain the firm’s competitive advantage in the market. To do so, you’ll enhance the existing trading signals with machine learning algorithms that can adapt to new data.
# 
# ## Instructions:
# 
# Use the starter code file to complete the steps that the instructions outline. The steps for this Challenge are divided into the following sections:
# 
# * Establish a Baseline Performance
# 
# * Tune the Baseline Trading Algorithm
# 
# * Evaluate a New Machine Learning Classifier
# 
# * Create an Evaluation Report
# 
# #### Establish a Baseline Performance
# 
# In this section, you’ll run the provided starter code to establish a baseline performance for the trading algorithm. To do so, complete the following steps.
# 
# Open the Jupyter notebook. Restart the kernel, run the provided cells that correspond with the first three steps, and then proceed to step four. 
# 
# 1. Import the OHLCV dataset into a Pandas DataFrame.
# 
# 2. Generate trading signals using short- and long-window SMA values. 
# 
# 3. Split the data into training and testing datasets.
# 
# 4. Use the `SVC` classifier model from SKLearn's support vector machine (SVM) learning method to fit the training data and make predictions based on the testing data. Review the predictions.
# 
# 5. Review the classification report associated with the `SVC` model predictions. 
# 
# 6. Create a predictions DataFrame that contains columns for “Predicted” values, “Actual Returns”, and “Strategy Returns”.
# 
# 7. Create a cumulative return plot that shows the actual returns vs. the strategy returns. Save a PNG image of this plot. This will serve as a baseline against which to compare the effects of tuning the trading algorithm.
# 
# 8. Write your conclusions about the performance of the baseline trading algorithm in the `README.md` file that’s associated with your GitHub repository. Support your findings by using the PNG image that you saved in the previous step.
# 
# #### Tune the Baseline Trading Algorithm
# 
# In this section, you’ll tune, or adjust, the model’s input features to find the parameters that result in the best trading outcomes. (You’ll choose the best by comparing the cumulative products of the strategy returns.) To do so, complete the following steps:
# 
# 1. Tune the training algorithm by adjusting the size of the training dataset. To do so, slice your data into different periods. Rerun the notebook with the updated parameters, and record the results in your `README.md` file. Answer the following question: What impact resulted from increasing or decreasing the training window?
# 
# > **Hint** To adjust the size of the training dataset, you can use a different `DateOffset` value&mdash;for example, six months. Be aware that changing the size of the training dataset also affects the size of the testing dataset.
# 
# 2. Tune the trading algorithm by adjusting the SMA input features. Adjust one or both of the windows for the algorithm. Rerun the notebook with the updated parameters, and record the results in your `README.md` file. Answer the following question: What impact resulted from increasing or decreasing either or both of the SMA windows?
# 
# 3. Choose the set of parameters that best improved the trading algorithm returns. Save a PNG image of the cumulative product of the actual returns vs. the strategy returns, and document your conclusion in your `README.md` file.
# 
# #### Evaluate a New Machine Learning Classifier
# 
# In this section, you’ll use the original parameters that the starter code provided. But, you’ll apply them to the performance of a second machine learning model. To do so, complete the following steps:
# 
# 1. Import a new classifier, such as `AdaBoost`, `DecisionTreeClassifier`, or `LogisticRegression`. (For the full list of classifiers, refer to the [Supervised learning page](https://scikit-learn.org/stable/supervised_learning.html) in the scikit-learn documentation.)
# 
# 2. Using the original training data as the baseline model, fit another model with the new classifier.
# 
# 3. Backtest the new model to evaluate its performance. Save a PNG image of the cumulative product of the actual returns vs. the strategy returns for this updated trading algorithm, and write your conclusions in your `README.md` file. Answer the following questions: Did this new model perform better or worse than the provided baseline model? Did this new model perform better or worse than your tuned trading algorithm?
# 
# #### Create an Evaluation Report
# 
# In the previous sections, you updated your `README.md` file with your conclusions. To accomplish this section, you need to add a summary evaluation report at the end of the `README.md` file. For this report, express your final conclusions and analysis. Support your findings by using the PNG images that you created.
# 

# In[112]:


# Imports
import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report


# ---
# 
# ## Establish a Baseline Performance
# 
# In this section, you’ll run the provided starter code to establish a baseline performance for the trading algorithm. To do so, complete the following steps.
# 
# Open the Jupyter notebook. Restart the kernel, run the provided cells that correspond with the first three steps, and then proceed to step four. 
# 

# ### Step 1: mport the OHLCV dataset into a Pandas DataFrame.

# In[113]:


# Import the OHLCV dataset into a Pandas Dataframe
ohlcv_df = pd.read_csv(
    Path("./Resources/emerging_markets_ohlcv.csv"), 
    index_col='date', 
    infer_datetime_format=True, 
    parse_dates=True
)

# Review the DataFrame
ohlcv_df.head()


# In[114]:


# Filter the date index and close columns
signals_df = ohlcv_df.loc[:, ["close"]]

# Use the pct_change function to generate  returns from close prices
signals_df["Actual Returns"] = signals_df["close"].pct_change()

# Drop all NaN values from the DataFrame
signals_df = signals_df.dropna()

# Review the DataFrame
display(signals_df.head())
display(signals_df.tail())


# ## Step 2: Generate trading signals using short- and long-window SMA values. 

# In[115]:


# Set the short window and long window
short_window = 4
long_window = 100

# Generate the fast and slow simple moving averages (4 and 100 days, respectively)
signals_df['SMA_Fast'] = signals_df['close'].rolling(window=short_window).mean()
signals_df['SMA_Slow'] = signals_df['close'].rolling(window=long_window).mean()

signals_df = signals_df.dropna()

# Review the DataFrame
display(signals_df.head())
display(signals_df.tail())


# In[116]:


# Initialize the new Signal column
signals_df['Signal'] = 0.0

# When Actual Returns are greater than or equal to 0, generate signal to buy stock long
signals_df.loc[(signals_df['Actual Returns'] >= 0), 'Signal'] = 1

# When Actual Returns are less than 0, generate signal to sell stock short
signals_df.loc[(signals_df['Actual Returns'] < 0), 'Signal'] = -1

# Review the DataFrame
display(signals_df.head())
display(signals_df.tail())


# In[117]:


signals_df['Signal'].value_counts()


# In[118]:


# Calculate the strategy returns and add them to the signals_df DataFrame
signals_df['Strategy Returns'] = signals_df['Actual Returns'] * signals_df['Signal'].shift()

# Review the DataFrame
display(signals_df.head())
display(signals_df.tail())


# In[119]:


# Plot Strategy Returns to examine performance
(1 + signals_df['Strategy Returns']).cumprod().plot()


# ### Step 3: Split the data into training and testing datasets.

# In[120]:


# Assign a copy of the sma_fast and sma_slow columns to a features DataFrame called X
X = signals_df[['SMA_Fast', 'SMA_Slow']].shift().dropna()

# Review the DataFrame
X.head()


# In[121]:


# Create the target set selecting the Signal column and assigning it to y
y = signals_df['Signal']

# Review the value counts
y.value_counts()


# In[122]:


# Select the start of the training period
training_begin = X.index.min()

# Display the training begin date
print(training_begin)


# In[123]:


# Select the ending period for the training data with an offset of 3 months
training_end = X.index.min() + DateOffset(months=3)

# Display the training end date
print(training_end)


# In[124]:


# Generate the X_train and y_train DataFrames
X_train = X.loc[training_begin:training_end]
y_train = y.loc[training_begin:training_end]

# Review the X_train DataFrame
X_train.head()


# In[125]:


# Generate the X_test and y_test DataFrames
X_test = X.loc[training_end+DateOffset(hours=1):]
y_test = y.loc[training_end+DateOffset(hours=1):]

# Review the X_test DataFrame
X_train.head()


# In[126]:


# Scale the features DataFrames

# Create a StandardScaler instance
scaler = StandardScaler()

# Apply the scaler model to fit the X-train data
X_scaler = scaler.fit(X_train)

# Transform the X_train and X_test DataFrames using the X_scaler
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# ### Step 4: Use the `SVC` classifier model from SKLearn's support vector machine (SVM) learning method to fit the training data and make predictions based on the testing data. Review the predictions.

# In[127]:


# From SVM, instantiate SVC classifier model instance
svm_model = svm.SVC()
 
# Fit the model to the data using the training data
svm_model = svm_model.fit(X_train_scaled, y_train)
 
# Use the testing data to make the model predictions
svm_pred = svm_model.predict(X_train_scaled)

# Review the model's predicted values
svm_pred


# ### Step 5: Review the classification report associated with the `SVC` model predictions. 

# In[128]:


# Use a classification report to evaluate the model using the predictions and testing data
svm_testing_report = classification_report(y_train, svm_pred)

# Print the classification report
print(svm_testing_report)


# ### Step 6: Create a predictions DataFrame that contains columns for “Predicted” values, “Actual Returns”, and “Strategy Returns”.

# In[129]:


# Create a new empty predictions DataFrame.

# Create a predictions DataFrame
predictions_df = pd.DataFrame(index=X_train.index)

# Add the SVM model predictions to the DataFrame
predictions_df['Predicted'] = svm_pred

# Add the actual returns to the DataFrame
predictions_df['Actual Returns'] = signals_df["Actual Returns"]

# Add the strategy returns to the DataFrame
predictions_df['Strategy Returns'] = predictions_df['Actual Returns'].copy() * predictions_df['Predicted']

# Review the DataFrame
display(predictions_df.head())
display(predictions_df.tail())


# ### Step 7: Create a cumulative return plot that shows the actual returns vs. the strategy returns. Save a PNG image of this plot. This will serve as a baseline against which to compare the effects of tuning the trading algorithm.

# In[130]:


# Plot the actual returns versus the strategy returns
svm_plot = (1 + predictions_df[["Actual Returns", "Strategy Returns"]]).cumprod().plot()

svm_plot


# ---
# 
# ## Tune the Baseline Trading Algorithm

# ## Step 6: Use an Alternative ML Model and Evaluate Strategy Returns

# In this section, you’ll tune, or adjust, the model’s input features to find the parameters that result in the best trading outcomes. You’ll choose the best by comparing the cumulative products of the strategy returns.

# ### Step 1: Tune the training algorithm by adjusting the size of the training dataset. 
# 
# To do so, slice your data into different periods. Rerun the notebook with the updated parameters, and record the results in your `README.md` file. 
# 
# Answer the following question: What impact resulted from increasing or decreasing the training window?

# Increasing the training window from 3 to 6 months caused the model to perform worse.

# ### Step 2: Tune the trading algorithm by adjusting the SMA input features. 
# 
# Adjust one or both of the windows for the algorithm. Rerun the notebook with the updated parameters, and record the results in your `README.md` file. 
# 
# Answer the following question: What impact resulted from increasing or decreasing either or both of the SMA windows?

# Increasing the sma short window from 4 to 14 days resulted in no action taken by the model.

# ### Step 3: Choose the set of parameters that best improved the trading algorithm returns. 
# 
# Save a PNG image of the cumulative product of the actual returns vs. the strategy returns, and document your conclusion in your `README.md` file.

# The original parameters for the svm model was the best approach. Given the data, it seems the the shorter outlook of the model the better. Although, we may find there is a lower bound on timeframe as well... more testing would be needed to evaluate all parameters.

# ---
# 
# ## Evaluate a New Machine Learning Classifier
# 
# In this section, you’ll use the original parameters that the starter code provided. But, you’ll apply them to the performance of a second machine learning model. 

# ### Step 1:  Import a new classifier, such as `AdaBoost`, `DecisionTreeClassifier`, or `LogisticRegression`. (For the full list of classifiers, refer to the [Supervised learning page](https://scikit-learn.org/stable/supervised_learning.html) in the scikit-learn documentation.)

# In[131]:


# Import a new classifier from SKLearn
from sklearn.ensemble import AdaBoostClassifier

# Initiate the model instance
ada = AdaBoostClassifier()


# ### Step 2: Using the original training data as the baseline model, fit another model with the new classifier.

# In[132]:


# Fit the model using the training data
ada = ada.fit(X_train_scaled, y_train)

# Use the testing dataset to generate the predictions for the new model
ada_pred = ada.predict(X_train_scaled)

# Review the model's predicted values
ada_pred


# ### Step 3: Backtest the new model to evaluate its performance. 
# 
# Save a PNG image of the cumulative product of the actual returns vs. the strategy returns for this updated trading algorithm, and write your conclusions in your `README.md` file. 
# 
# Answer the following questions: 
# Did this new model perform better or worse than the provided baseline model? 
# Did this new model perform better or worse than your tuned trading algorithm?

# In[133]:


# Use a classification report to evaluate the model using the predictions and testing data
ada_report = classification_report(y_train, ada_pred)

# Print the classification report
print(ada_report)


# In[134]:


# Create a new empty predictions DataFrame.

# Create a predictions DataFrame
ada_predictions_df = pd.DataFrame(index=X_train.index)

# Add the SVM model predictions to the DataFrame
ada_predictions_df['Predicted'] = ada_pred

# Add the actual returns to the DataFrame
ada_predictions_df['Actual Returns'] = signals_df["Actual Returns"]

# Add the strategy returns to the DataFrame
ada_predictions_df['Strategy Returns'] = ada_predictions_df['Actual Returns'].copy() * ada_predictions_df['Predicted']

# Review the DataFrame
display(ada_predictions_df.head())
display(ada_predictions_df.tail())


# In[135]:


# Plot the actual returns versus the strategy returns
ada_plot = (1 + ada_predictions_df[["Actual Returns", "Strategy Returns"]]).cumprod().plot()

ada_plot

