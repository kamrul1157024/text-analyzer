import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


class BanglaMlModelReader:

    @staticmethod
    def getModels(tfIDFPath, modelPath):
        with open(modelPath, "rb") as modelFile:
            model = pickle.load(modelFile)
        with open(tfIDFPath, "rb") as tfIDF:
            tfIDFFile = pickle.load(tfIDF)
        ngram_range = (1, 3)
        min_df = 10
        max_df = 1.00
        max_features = 200
        tfIDF = TfidfVectorizer(encoding='utf-8',
                                ngram_range=ngram_range,
                                stop_words=None,
                                lowercase=False,
                                max_df=max_df,
                                min_df=min_df,
                                max_features=max_features,
                                norm='l2',
                                vocabulary=tfIDFFile.vocabulary_,
                                sublinear_tf=True)

        return tfIDF, model


class EnglishMlModelReader:

    @staticmethod
    def getModels(tfIDFPath, modelPath):
        with open(modelPath, "rb") as modelFile:
            model = pickle.load(modelFile)
        with open(tfIDFPath, "rb") as tfIDF:
            tfIDFFile = pickle.load(tfIDF)

        ngram_range=(1,3)
        min_df=10
        max_df=1.00
        max_features=300
        tfIDF = TfidfVectorizer(encoding='utf-8',
                                ngram_range=ngram_range,
                                stop_words=None,
                                lowercase=False,
                                max_df=max_df,
                                min_df=min_df,
                                max_features=max_features,
                                norm='l2',
                                vocabulary=tfIDFFile.vocabulary_,
                                sublinear_tf=True)

        return tfIDF, model


class LanguageMlModelReader:

    @staticmethod
    def getModels(tfIDFPath, modelPath):
        with open(modelPath, "rb") as modelFile:
            model = pickle.load(modelFile)
        with open(tfIDFPath, "rb") as tfIDF:
            tfIDFFile = pickle.load(tfIDF)

        ngram_range = (1, 1)
        min_df = 10
        max_df = 1.00
        max_features = 100
        tfIDF = TfidfVectorizer(encoding='utf-8',
                                ngram_range=ngram_range,
                                stop_words=None,
                                lowercase=False,
                                max_df=max_df,
                                min_df=min_df,
                                max_features=max_features,
                                norm='l2',
                                vocabulary=tfIDFFile.vocabulary_,
                                sublinear_tf=True)

        return tfIDF, model
