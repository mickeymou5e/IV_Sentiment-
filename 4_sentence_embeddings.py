#Kill Me PLS :)

import seaborn as sns
import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
import time
import functools as ft
import warnings
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
from sklearn.metrics import accuracy_score
import scikitplot as skplt #pip install scikit-plot
import matplotlib.pyplot as plt

class PredicitionObj:
    def __init__(self,df):
        self.df = df

    def encodingTweets(self, columnNameTweetsForEncoding,newColumnNameForEncodedTweets):
        start = time.time()

        sbert_model = SentenceTransformer('stsb-mpnet-base-v2')
        self.df[newColumnNameForEncodedTweets] = self.df[columnNameTweetsForEncoding].apply(lambda x: sbert_model.encode(x, show_progress_bar=True))

        end = time.time()
        print(end - start)

        #This attribute below is only for BERT if you are using your own Ngrams you will need to change it, shouldn't be too hard tho
        self.dfVectorizedTweetsAsFeatures = pd.DataFrame(self.df[newColumnNameForEncodedTweets].to_list(), columns=list(range(1, 769)))

    def trainingGradientBoostingModel(self,dataframeContainingOnlytheY_Variable):

        warnings.filterwarnings("ignore")

        self.X, self.y = self.dfVectorizedTweetsAsFeatures, dataframeContainingOnlytheY_Variable

        cats = self.X.select_dtypes(exclude=np.number).columns.tolist()  # Categorical Encoding, not Neceasry but if we add features in the future which are non numeric

        for col in cats:

            self.X[col] = self.X[col].astype(
                'category')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=1, shuffle=False)

        self.dtrain_clf = xgb.DMatrix(self.X_train, self.y_train)
        self.dtest_clf = xgb.DMatrix(self.X_test, self.y_test)


    def CrossValidationOfModel(self):
        n = 1000

        params = {"objective": "binary:logistic", "tree_method": "hist"}

        self.results = xgb.cv(

            params, self.dtrain_clf,

            num_boost_round=n,

            nfold=5,

            metrics=["auc"]

        )
        self.aucMaxValue = self.results['test-auc-mean'].max()


    def trainingAndTestingOfModel(self):
        n = 1000

        params = {"objective": "binary:logistic", "tree_method": "hist"}

        self.model = xgb.train(

            params=params,

            dtrain=self.dtrain_clf,

            num_boost_round=n,

        )

        self.predictionsFromModelAsProbabilities = self.model.predict(self.dtest_clf)

        self.predictionsFromModelCategorized = [round(value) for value in self.predictionsFromModelAsProbabilities]

        self.simpleAccuracyOfModel = accuracy_score(self.y_test, self.predictionsFromModelCategorized)

    def plottingOfTheROC(self):

        self.confusionMatrixModel = metrics.confusion_matrix(self.y_test.to_numpy(),self.predictionsFromModelCategorized)
        self.cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=self.confusionMatrixModel ,
                                                    display_labels=["No Movement", "Movement"])
        self.cm_display.plot()
        plt.show()

fileWithTweets = "AAPL_4.csv" #name of file with tweets
DF = pd.read_csv(fileWithTweets)




ObjectPred = PredicitionObj(DF)
ObjectPred.encodingTweets("Tweets","newName")
ObjectPred.trainingGradientBoostingModel(DF["Changed"]) #This is just the predicted values, this can be anything, if you have it in one file thats fine just specify the df and the column
ObjectPred.CrossValidationOfModel()
ObjectPred.trainingAndTestingOfModel()
ObjectPred.simpleAccuracyOfModel
ObjectPred.aucMaxValue
ObjectPred.plottingOfTheROC()

ObjectPred.y_test = pd.DataFrame(ObjectPred.y_test)
(ObjectPred.y_test)["Pred as Prob"] = np.array(ObjectPred.predictionsFromModelAsProbabilities)
(ObjectPred.y_test)["Pred as Categories"] = np.array(ObjectPred.predictionsFromModelCategorized)

ObjectPred.y_test.to_csv("Patrik" + fileWithTweets)
