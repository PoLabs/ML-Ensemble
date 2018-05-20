# ML Ensemble


Uses a greedy linear ensemble of 7 models to predict changes at 5-30mins out. 
The 'caret' wrapper is used for grid searching and 'caretEnsemble' for the ensemble. 
Models: 
- glm
- Bayes glm
- rpart
- random forest
- support vector machine
- xgboost for gradient boosted trees
- mxnet neural network

Data pulled directly from Binance API. 
