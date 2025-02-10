"""The main scraping script to get Syriac texts from CAL."""

from bs4 import BeautifulSoup
from cal_handler import pool_init, get_a_chapter, follow_link
# from pathlib import Path
import re
import json


http = pool_init()
result = get_a_chapter(http)

soup = BeautifulSoup(result, 'lxml')

lines = []
word_lis = []

for table_data in soup.find_all('td'):
    for link in table_data.find_all('a'):

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
                        # remove numbering appended to lemma, like "#2"
                        # and replace "@" in lemma with a space
                        res = re.sub("#\d", "", res.split(' ')[0]).replace("@", " ")
                        word_lis.append(res)

    # when the scraper reaches a new row on the table
    if len(word_lis) > 0:
        print(word_lis)
        lines.append(word_lis)
        word_lis = []

# print(lines)
# Dump the scraped lines into a json file
with open("scraper_results.json", "w+") as sr:
    json.dump(lines, sr)

print("Process Complete!")