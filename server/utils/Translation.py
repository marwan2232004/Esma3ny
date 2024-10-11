import requests
import sys
import io


def get_translate(article_ar):
    url = "https://openl-translate.p.rapidapi.com/translate"

    payload = {"target_lang": "arz", "text": article_ar}
    headers = {
        "x-rapidapi-key": "394717c44cmsh82600cc7cdcfb18p1008a8jsne46053c531d3",
        "x-rapidapi-host": "openl-translate.p.rapidapi.com",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()["translatedText"]


# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
# print(get_translate("Hello mohamed I want to go kitchen and eating chicken with your mom"))
