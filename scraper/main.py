"""The main scraping script to get Syriac texts from CAL."""

from bs4 import BeautifulSoup, Tag
from cal_handler import pool_init, get_a_chapter, follow_link
# from pathlib import Path
import re
import json


def count_char(target: str, source: str, counter: int):
    for letter in source:
        if letter == target:
            counter = counter + 1

def is_lemma(lex: str):
    if "part of previous word" in lex or "non-Aramaic or fragmentary" in lex:
        return False
    return True

def find_lemma(markup_tag: Tag):
    """Find the lemma from given HTML excerpt.
    
    markup_tag: Tag
        result(s) returned by bs4's find() or find_all()
    """
    if markup_tag.name is None:
        # if the content of body tag is not within any tag
        # but a raw string
        stripped = markup_tag.text.strip()
        # remove trailing periods lying outside the tags
        res = re.sub("^\\.$", "", stripped)
        if res != "" and is_lemma(res):
            # get the lemma and remove POS description etc.
            res = res.split(' ')[0]
            # remove numbering appended to lemma, like "???#2"
            # and replace "@" in compound words with a space
            return re.sub("#\\d", "", res).replace("@", " ")


if __name__ == "__main__":
    http = pool_init()
    result = get_a_chapter(http)

    # Use lxml parser to correctly handle raw texts in body tag
    soup = BeautifulSoup(result, 'lxml')

    # lists to store results
    verses = []
    word_lis = []

    # counters
    num_slashes = 0
    num_words = 0

    for table_data in soup.find_all('td'):
        for link in table_data.find_all('a'):

            lex_url = link["href"]

            if "getlex" in lex_url:
                # if it's linked to getlex.php file, it's a word
                count_char("/", link.text, num_slashes)  # count the number of slash

                # follow the link to get the lemma(ta) page
                lex_page = follow_link(http, lex_url)
                lex_soup = BeautifulSoup(lex_page, 'lxml')
                lex_body = lex_soup.body

                for body_child in lex_body.children:
                    res = find_lemma(body_child)
                    if res is not None:
                        word_lis.append(res)

        # when the scraper reaches a new row on the table,
        # it's a new verse
        if len(word_lis) > 0:
            print(word_lis)
            num_words = num_words + len(word_lis)
            verses.append(word_lis)
            word_lis = []
        
    # the number of scribal variances is number of slashes divided by 2
    num_variances = num_slashes / 2

    # Dump the scraped lines into a json file
    with open("scraper_results.json", "w+") as sr:
        json.dump(verses, sr)

    print("Process Complete!")
    print("number of words: {}".format(num_words))
    print("number of scribal variants: {}".format(num_variances))