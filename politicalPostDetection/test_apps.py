from unittest import TestCase
from politicalPostDetection.MLModelReader import BanglaMlModelReader, EnglishMlModelReader, LanguageMlModelReader
from .apps import PoliticalPostDetectionConfig as pcfg
from .MLModelPredictor import MLModelPredictor

class TestPoliticalPostDetectionConfig(TestCase):
    def setUp(self):
        self.bangla_tfIDF, self.bangla_gaussianNaiveBiasModel = BanglaMlModelReader.getModels(pcfg.bangla_tfIDFPath,
                                                                                              pcfg.bangla_GaussianNaiveBiasPath)

        self.english_tfIDF, self.english_SVMSigmoid = EnglishMlModelReader.getModels(pcfg.english_tfIDFPath,
                                                                                     pcfg.english_SVMSigmoidPath)

        self.language_tfIDF, self.language_GaussianNaiveBias = LanguageMlModelReader.getModels(pcfg.language_tfIDFPath,
                                                                                               pcfg.language_GaussianNaiveBiasPath)

    def testModelNames(self):
        self.assertEqual(type(self.bangla_gaussianNaiveBiasModel).__name__, "GaussianNB")
        self.assertEqual(type(self.english_SVMSigmoid).__name__, "SVC")
        self.assertEqual(type(self.language_GaussianNaiveBias).__name__, "MultinomialNB")


    def testNameShouldBeTfIDF(self):
        self.assertEqual(type(self.bangla_tfIDF).__name__, "TfidfVectorizer")
        self.assertEqual(type(self.english_tfIDF).__name__, "TfidfVectorizer")
        self.assertEqual(type(self.language_tfIDF).__name__, "TfidfVectorizer")

    def testBanglaModelPrediction(self):
        text="আওয়ামী লীগ কখনোই জনগণের বন্ধু ছিল না—এমন মন্তব্য করে বিএনপির মহাসচিব মির্জা ফখরুল ইসলাম আলমগীর বলেছেন, ‘পাকিস্তান আমলে আওয়ামী লীগ একসময় আমাদের সঙ্গে কাঁধে কাঁধ মিলিয়ে গণতন্ত্রের সংগ্রাম করেছে। অথচ স্বাধীনতার পর আওয়ামী লীগের চরিত্র সম্পূর্ণ বদলে গেছে। তারা সেই গণতন্ত্রের মধ্যে আর নিজেদের ধারণ করতে পারছে না। কারণ, গণতন্ত্র থাকলে ক্ষমতা চলে যাওয়ার সম্ভাবনা থাকে।’আজ শনিবার দুপুরে জাতীয় প্রেসক্লাবে এক কর্মসূচিতে অংশ নিয়ে মির্জা ফখরুল এসব কথা বলেন। বিনা মূল্যে চিকিৎসাসেবা, ওষুধ বিতরণ ও স্বেচ্ছায় রক্তদানের এই কর্মসূচির আয়োজন করে স্বাধীনতার সুবর্ণজয়ন্তী উদ্‌যাপনে বিএনপির গঠিত চিকিৎসা ও সেবা কমিটি। এই কর্মসূচির উদ্বোধন করেন মির্জা ফখরুল।এতে বিশেষ অতিথি ছিলেন দলের স্থায়ী কমিটির সদস্য খন্দকার মোশাররফ হোসেন। আরও বক্তব্য দেন দলের স্থায়ী কমিটির আরেক সদস্য মির্জা আব্বাস, বিএনপির চেয়ারপারসনের উপদেষ্টা আবদুস সালাম প্রমুখ।িএনপির মহাসচিব বলেন, ‘গণতন্ত্র থাকলে সবাইকে ভিন্নমত প্রকাশের স্বাধীনতা দিতে হবে। গণতন্ত্র মানলে ন্যায়বিচার ও সুশাসন প্রতিষ্ঠা করতে হবে। এগুলোর মধ্যে আওয়ামী লীগ নেই। তারাই সব। তারাই মালিক। আমরা সব প্রজা। এভাবেই তারা গোটা বাংলাদেশকে দেখে।’"
        print(MLModelPredictor().predictBangla(text))