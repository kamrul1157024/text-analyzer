from django.apps import AppConfig
from pathlib import Path

from .MLModelReader import \
    BanglaMlModelReader, EnglishMlModelReader, LanguageMlModelReader

import pandas as pd
import nltk

nltk.download('punkt')
nltk.download('stopwords')


class PoliticalPostDetectionConfig(AppConfig):
    name = 'politicalPostDetection'

    bangla_GaussianNaiveBiasPath = Path("politicalPostDetection/banglaMLModels/gaussian_naive_bias.pickle")
    bangla_tfIDFPath = Path("politicalPostDetection/banglaMLModels/tfidf_bangla_pickle.pkl")

    english_SVMSigmoidPath = Path("politicalPostDetection/englishMLModels/svm_sigmoid.pickle")
    english_tfIDFPath = Path("politicalPostDetection/englishMLModels/tfidf_english_pickle.pkl")

    language_GaussianNaiveBiasPath = Path("politicalPostDetection/languageMLModel/multinomial_naive_bias.pickle")
    language_tfIDFPath = Path("politicalPostDetection/languageMLModel/tfidf_lang_recoginition.pkl")

    bangla_tfIDF, bangla_gaussianNaiveBiasModel = BanglaMlModelReader.getModels(bangla_tfIDFPath,
                                                                                bangla_GaussianNaiveBiasPath)

    english_tfIDF, english_SVMSigmoid = EnglishMlModelReader.getModels(english_tfIDFPath, english_SVMSigmoidPath)

    language_tfIDF, language_MultinomialNaiveBias = LanguageMlModelReader.getModels(language_tfIDFPath,
                                                                                    language_GaussianNaiveBiasPath)

    bangla_stopwords = pd.read_csv('politicalPostDetection/banglaMLModels/bangla_stopwords.txt', header=None)[0]
    banglaStopWordsMap = {}
    for w in bangla_stopwords:
        banglaStopWordsMap[w] = True
