from .TextProcessor import BanglaTextPreprocessor, EnglishTextPreprocessor
from .apps import PoliticalPostDetectionConfig as pcfg
from .LanguageProbability import LanguageProbability


class MLModelPredictor:

    def __init__(self):
        self.__banglaTextPreprocessor = BanglaTextPreprocessor()
        self.__englishTextPreprocessor = EnglishTextPreprocessor()
        self.__banglaModel = pcfg.bangla_gaussianNaiveBiasModel
        self.__banglaTfIDF = pcfg.bangla_tfIDF
        self.__englishModel = pcfg.english_SVMSigmoid
        self.__englishTfIDF = pcfg.english_tfIDF
        self.__languageModel = pcfg.language_MultinomialNaiveBias
        self.__languageTfIDF = pcfg.language_tfIDF

    def predictBangla(self, banglaText):
        banglaText = self.__banglaTextPreprocessor.process_text(banglaText)
        text_for_pred = self.__banglaTfIDF.fit_transform([banglaText]).toarray()
        model_predictions = self.__banglaModel.predict_proba(text_for_pred)
        class_probability = model_predictions.flatten()
        return class_probability[1]

    def predictEnglish(self, englishText):
        englishText = self.__englishTextPreprocessor.process_text(englishText)
        text_for_pred = self.__englishTfIDF.fit_transform([englishText]).toarray()
        model_predictions = self.__englishModel.predict_proba(text_for_pred)
        class_probability = model_predictions.flatten()
        return class_probability[1]

    def predictLanguage(self, text):
        text_for_pred = self.__languageTfIDF.fit_transform([text]).toarray()
        model_predictions = self.__languageModel.predict_proba(text_for_pred)
        class_probability = model_predictions.flatten()
        languageProbability = LanguageProbability(class_probability)
        return languageProbability

    def predictPostPoliticalProbability(self, text):
        fullText = text['fullText']
        banglaText = text['bangla']
        englishText = text['english']
        banglishText = text['banglish']

        # [bangla banglish english]
        languageProbability = self.predictLanguage(fullText)
        # [notPolitical political]
        banglaProbability = self.predictBangla(banglaText)
        # [notPolitical political]
        englishProbability = self.predictEnglish(englishText)
        # [notPolitical political]
        banglishProbability = self.predictBangla(banglishText)

        probabilityOfBeingPolitical = ((languageProbability.bangla * banglaProbability)
                                       + (languageProbability.banglish * banglishProbability)
                                       + languageProbability.english * englishProbability)

        return probabilityOfBeingPolitical, languageProbability, banglaProbability, banglishProbability, \
               englishProbability;
