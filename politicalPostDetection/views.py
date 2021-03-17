from rest_framework.decorators import api_view
from rest_framework.response import Response
from .MLModelPredictor import MLModelPredictor


# requestPararms
#
# {
#     "fullText":".............",
#     "bangla":"...........",
#     "banglish":"........",
#     "english":"........."
# }

@api_view(["POST"])
def isPoliticalPost(request):
    text = request.data
    mlModelPredictor = MLModelPredictor()
    probabilityOfBeingPolitical, \
    languageProbability, \
    banglaProbability, \
    banglishProbability, \
    englishProbability = mlModelPredictor.predictPostPoliticalProbability(text)

    responseData = {
        "languageProbability": languageProbability.getJSON(),
        "probabilityOfBeingPolitical":probabilityOfBeingPolitical,
        "banglaProbability": banglaProbability,
        "banglishProbability": banglishProbability,
        "englishProbability": englishProbability
    }
    return Response(responseData)


@api_view(["POST"])
def detectLanguage(request):
    text = request.data["text"]
    mlModelPredictor = MLModelPredictor()
    languageProbability= mlModelPredictor.predictLanguage(text)

    return Response(languageProbability.getJSON())
