"""The main scraping script to get Syriac texts from CAL."""

from bs4 import BeautifulSoup
from cal_handler import pool_init, get_a_chapter
# from pathlib import Path
import re


def clean_word(text: str):
    # TODO: deal with things like:
    # wpr:$)/w)p
    # pr:$)#3#/,

    # TODO: deal with things like:
    # \slqw
    # (mh
    # /#3#/,

    if "#" in text and "<" in text:  # If the word is surrounded by "< >"
        text = text.replace("<").replace(">")
    elif "#" in text and "/" in text:
        text = re.sub("/.+/", "")
    text = re.sub("#.+#", text.strip())  # remove "#3#"


http = pool_init()
results = get_a_chapter(http)

soup = BeautifulSoup(results, 'html.parser')

lines = []
for link in soup.find_all('a'):
    lis = []
    if "getlex" in link["href"]:
        word = clean_word(link.text)
        lis.append(link.text)
    else:
        lines.append(lis)

print("Process Complete!")