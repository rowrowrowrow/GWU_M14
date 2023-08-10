# Module 14: Algorithmic Trading

## Overview

This repo demonstrates algorithmic trading using predefined conditional logic and machine learning. It compares various techniques while exploring the effect of changes in variable inputs.

## Results

* Model 1:
```
              precision    recall  f1-score   support

        -1.0       1.00      0.06      0.12        49
         1.0       0.63      1.00      0.77        79

    accuracy                           0.64       128
   macro avg       0.82      0.53      0.44       128
weighted avg       0.77      0.64      0.52       128
```

* Model 2 (6 month training):
```
              precision    recall  f1-score   support

        -1.0       0.59      0.20      0.29       122
         1.0       0.58      0.89      0.70       154

    accuracy                           0.58       276
   macro avg       0.58      0.54      0.50       276
weighted avg       0.58      0.58      0.52       276
```

* Model 3 (14 day SMA):
```
              precision    recall  f1-score   support

        -1.0       0.00      0.00      0.00        49
         1.0       0.62      1.00      0.76        79

    accuracy                           0.62       128
   macro avg       0.31      0.50      0.38       128
weighted avg       0.38      0.62      0.47       128
```

* Model 4 (ADA Boost Machine Learning):
```
              precision    recall  f1-score   support

        -1.0       0.97      0.65      0.78        49
         1.0       0.82      0.99      0.90        79

    accuracy                           0.86       128
   macro avg       0.90      0.82      0.84       128
weighted avg       0.88      0.86      0.85       128
```

## Summary

Increasing the training window from 3 to 6 months caused the model to perform worse.

Increasing the sma short window from 4 to 14 days resulted in no action taken by the model.

The original parameters for the svm model was the best approach. Given the data, it seems the the shorter outlook of the model the better. Although, we may find there is a lower bound on timeframe as well... more testing would be needed to evaluate all parameters.

---

## Technologies

This application uses python 3, please install the necessary packages as described below to recreate the analysis.

---

## Installation Guide

```
pip install -r requirements.txt
```

---

## Usage

Several notebooks are contained within this repo, each for the model that was tested. The two best are included in `machine_learning_trading_bot.ipynb`. Data sources are located in the `Resources` folder.

Second best (Model 1):

[rowrowrowrow](original_actual_vs_strategy.png)


Best (Model 4):

[rowrowrowrow](ada_boost_strategy.png)

---

## Contributors

[rowrowrowrow](https://github.com/rowrowrowrow)

---

## License

No license provided, you may not use the contents of this repo.
