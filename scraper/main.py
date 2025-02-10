"""The main scraping script to get Syriac texts from CAL."""

from bs4 import BeautifulSoup
from cal_handler import pool_init, get_a_chapter, follow_link
# from pathlib import Path
import re


def clean_word(text: str, flags: dict) -> str:
    # Done: TODO: deal with things like:
    # wpr:$)/w)p
    # pr:$)#3#/,

    # Done: TODO: deal with things like:
    # \slqw
    # (mh
    # /#3#/,

    # Initial cleaning
    text = text.strip()

    if "#" in text and "<" in text:
        # If the word is surrounded by "< >"
        text = text.replace("<", "").replace(">", "")
    
    if flags["in_variant"] and "/" in text:
        # second "/" marking the end of variant spelling,
        # remove the text before this "/"
        text = re.sub(".+/", "", text)
        flags["in_variant"] = False
    elif flags["in_variant"]:
        # inside the variant spelling format, ignore this word
        text = ""
    elif "#" in text and "/" in text:
        # If the word contains a variant spelling formatted like "???/???#3#/",
        # Remove it
        text = re.sub("/.+/", "", text)
    elif "/" in text:
        # beginning of a variant spelling, where it's one word in the revised text
        # but it's multiple words in the referenced alternative source
        flags["in_variant"] = True
    text = re.sub("#.+#", "", text)  # remove "#3#"
    text = text.replace("\\", "")  # remove "\"

    return text


http = pool_init()
result = get_a_chapter(http)

soup = BeautifulSoup(result, 'html.parser')

lines = []
flag_dict = {
    "in_variant": False
}
word_lis = []

for link in soup.find_all('a'):
    # TODO: follow the links
    word = ""
    if "getlex" in link["href"]:
        # if it's linked to getlex.php file, it's a word
        # follow the link to get the lemma(ta) page
        lex_page = follow_link(http, link["href"])
        lex_soup = BeautifulSoup(lex_page, "html.parser")
        lemmata = lex_soup.find_all("span", class_="lem")
        for lemma in lemmata:
            lemma_variants = lemma.text.split(", ")  # variations of the same lemma
            lemma_one = lemma_variants[0]
            print(lemma_one)
            word_lis.append(lemma_one)        
    else:
        lines.append(word_lis)
        word_lis = []

print("Process Complete!")
print(lines)