# Tech Stock Option Implied Volatility Prediction using Tweets and BERT

This repository contains code for a machine learning model that predicts the implied volatility of options on technology stocks using tweets and BERT, a state-of-the-art language model developed by Google. 
## Introduction

The implied volatility of an option is a measure of the expected volatility of the underlying stock over the life of the option. It is a key input to options pricing models and is used to estimate the probability of the stock price reaching certain levels by the expiration date of the option.

Predicting implied volatility is a challenging task, as it depends on a range of factors such as market sentiment, news, and events related to the underlying stock. Twitter is a rich source of information about such factors, as it is widely used by traders, investors, and market participants to share their opinions and insights.

BERT is a powerful language model that is capable of understanding the meaning and context of natural language text, making it well-suited for tasks such as sentiment analysis and text classification.

In this project, we use BERT to analyse tweets related to technology stocks and predict the implied volatility of options on those stocks.

## Data

We use a dataset of 4 million tweets related to technology stocks, obtained from a Kaggle Dataset and Bloomberg. The tweets are labeled with the implied volatility of options (accessed from Bloomberg) on the corresponding stocks at the time the tweet was posted. We preprocess the tweets by removing links, images and emojis only. 

## Model

We use BERT as a feature extractor to extract meaningful representations of the tweets, which we then feed into a feedforward neural network to predict the implied volatility of the corresponding stock options. We train the model using a cross-entropy loss function.


## Requirements

To run the code in this repository, you will need Python 3.7 or higher, along with the following libraries:

- PyTorch
- Transformers (for BERT)
- Pandas
- Scikit-learn

## Disclosures


For full disclosure it takes days to build the embeddings, tensors etc. even training on VGPUS, feel free to contact me at patrik.kovac22@imperial.ac.uk to get access to our sentence embeddings and tensors to do your own analysis. 
