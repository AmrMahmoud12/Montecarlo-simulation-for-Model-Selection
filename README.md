# Montecarlo-simulation-for-Model-Selection

you can find the dataset here:
##  https://www.kaggle.com/mlg-ulb/creditcardfraud

In this notebook we apply a montecarlo based feature selection method to identify a reduced number of features that can be used as a good predictor for this credit card fraud dataset. Using a reduced number of features is of crucial importance when overfitting needs to be prevented. This models has been tested in some other datasets with similar results.  

We expand the set of explanatory features by computing the products of all features against the rest. When the product of two features has better sorting capabilities than both features individually we include the product in our set of candidate features, a minimum threshold is applied. 

Additionally the original set of features and the new set resulting from multiplying pairs of feature are transformed by means a logistic distribution. We have observed that this transformation increases predictive capabilities when compared to the ubiquitous normal transformation. 

In order to measure if a model has a good predictive power we define a modified Jaccard distance. As follows:
                                                  
$$Modified\,Jaccard\,Distance = 1 − \dfrac{\sum\limits_{i}{\min{ (target_{i},\,model\, probability_{i})}}}{\sum\limits_{i}{\max{(target_{i},\,model\, probability_{i})}}}$$                               
The lower the distance the best the model predicts the target.

In each Montecarlo iteration a reduced set of features, say 5 to 8, is randomly selected, then we compute the Logistic Regression model that best predicts fraud with this features and, finally we compute the modified Jaccard distance from the prediction to the target. The process is repeated for a large number of iterations. Resulting models are sorted by distance. 

We have tested modified Jaccard distance metric against most common metrics such as ROC, AUC, recall… and found that models with the low values of this modified Jaccard distance have a better balanced results in the rest of metrics. 

Final model selection is done by choosing the model with me minimum modified Jaccard distance, or any other among those with minimum distance that best fits the test subsample. 
