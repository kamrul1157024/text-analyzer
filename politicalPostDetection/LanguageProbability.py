class LanguageProbability:
    def __init__(self, languageProbability):
        self.bangla = languageProbability[0]
        self.banglish = languageProbability[1]
        self.english = languageProbability[2]

    def getJSON(self):
        return self.__dict__
