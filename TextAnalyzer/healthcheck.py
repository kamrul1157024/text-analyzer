from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view

@api_view(["GET"])
def getHealth(_):
    return Response('',status.HTTP_200_OK)
