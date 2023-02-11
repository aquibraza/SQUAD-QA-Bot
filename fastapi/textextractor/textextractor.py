
from typing import Text
import requests


class WikipediaDataExtractor():

    def __init__(self) -> None:
        self.targetUrl = "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles="


    def ExtractText(self, extractTitle):
        self.targetUrl = self.targetUrl + extractTitle
        extractedJson = requests.get(self.targetUrl)

        extractedText = list(extractedJson.json()['query']['pages'].keys())[0]
        extractedText = extractedJson.json()['query']['pages'][extractedText]["extract"]
        extractedText = extractedText.replace("\"", "")
        extractedText = extractedText.replace("\n", " ")
        extractedText = extractedText.replace('"', '')
        extractedText = extractedText.replace('=', '')
        extractedText = extractedText.replace("'", '')
        extractedText = extractedText.replace('\n', '').replace('\r', '')
        extractedText = extractedText.replace('\t', '')
        extractedText = extractedText.replace('-', ' ')
        extractedText = extractedText.replace('\r\n', '')

        return extractedText