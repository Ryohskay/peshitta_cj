"""The main scraping script to get Syriac texts from CAL."""

from bs4 import BeautifulSoup
from cal_handler import pool_init, get_a_chapter, follow_link
# from pathlib import Path
import re
import json

def clean_word(text: str, flags: dict) -> str:
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

soup = BeautifulSoup(result, 'lxml')

lines = []
flag_dict = {
    "in_variant": False,
    "in_multi_word_variant": False
}
word_lis = []

for link in soup.find_all('a'):

    # Remove the default variant spelling format "???/???#3#/"
    anchor_text = re.sub("/.+/", "", link.text.strip())

    # Set flags to control for variant spelling spanning multiple words
    if not flag_dict["in_variant"] and "/" in anchor_text:
        flag_dict["in_variant"] = True
    elif "/" in anchor_text:
        flag_dict["in_variant"] = False

    # Skip <a> tags if it's inside multi-word variance
    if flag_dict["in_variant"]:
        # deal with things like:
        # wpr:$)/w)p pr:$)#3#/,
        continue

    # TODO: deal with things like:
    # \slqw (mh /#3#/,

    # TODO: deal with things like:
    # \wlmk nsb_ lh /wnsb lh lmk#3#/

    lex_url = link["href"]

    if "getlex" in lex_url:
        # if it's linked to getlex.php file, it's a word
        # follow the link to get the lemma(ta) page
        lex_page = follow_link(http, lex_url)
        lex_soup = BeautifulSoup(lex_page, 'lxml')
        lex_body = lex_soup.body
        for body_child in lex_body.children:
            if body_child.name is None:
                # if the content of body tag is not within any tag
                # but a raw string
                stripped = body_child.text.strip()
                # remove trailing periods lying outside the tags
                res = re.sub("^\\.$", "", stripped)
                if res != "" and "part of previous word" not in res:
                    word_lis.append(res.split(' ')[0])
    elif len(word_lis) > 0:
        print(word_lis)
        lines.append(word_lis)
        word_lis = []

# print(lines)
with open("scraper_results.json", "w+") as sr:
    json.dump(lines, sr)

print("Process Complete!")