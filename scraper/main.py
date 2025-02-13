"""The main scraping script to get Syriac texts from CAL."""

from bs4 import BeautifulSoup, Tag
from cal_handler import pool_init, get_a_chapter, follow_link
from pathlib import Path
import re


def count_char(target: str, source: str) -> int:
    """Count the number of target characters."""
    num = 0
    for letter in source:
        if letter == target:
            num = num + 1
    return num

def is_lemma(lex: str) -> bool:
    """Returns if given string is a lemma in a CAL lexicon entry."""
    if "part of previous word" in lex or "non-Aramaic or fragmentary" in lex:
        # If the given string contains either of these phrases,
        # the entry does not list any lemma
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
    result = get_a_chapter(http, section=50)

    # Use lxml parser to correctly handle raw texts in body tag
    soup = BeautifulSoup(result, 'lxml')

    # lists to store results
    verses = []
    word_lis = []
    verse_ids = []

    # counters
    num_slashes = 0
    num_lemmata = 0
    num_links = 0

    for table_data in soup.find_all('td'):
        if "valign" in table_data.attrs.keys() and table_data["valign"] ==  "top":
            # when the scraper reaches a new row on the table.
            # push the scraped lemmata 
            if len(word_lis) > 0:
                print(word_lis)
                num_lemmata = num_lemmata + len(word_lis)
                verses.append(word_lis)
                word_lis = []
            
            # add the verse identifier (e.g. "01:01")
            verse_id = re.match("\\d\\d:\\d\\d", table_data.text)
            if verse_id is not None:
                vid = verse_id.group()
                print(vid)
                verse_ids.append(vid)
        else:  # If it's the cell containing verse
            # go through all links in the table data cell
            for link in table_data.find_all('a'):

                lex_url = link["href"]

                if "getlex" in lex_url:
                    # if it's linked to getlex.php file, it's a word
                    # count the number of slash
                    num_slashes = num_slashes + count_char("/", link.text)
                    # count the number of links
                    num_links = num_links + 1
                    # follow the link to get the lemma(ta) page
                    lex_page = follow_link(http, lex_url)
                    lex_soup = BeautifulSoup(lex_page, 'lxml')
                    lex_body = lex_soup.body

                    for body_child in lex_body.children:
                        res = find_lemma(body_child)
                        if res is not None:
                            word_lis.append(res)
        
    # the number of scribal variances is number of slashes divided by 2
    num_variances = num_slashes / 2

    if len(verses) != len(verse_ids):
        print("Something is wrong with verse ids vs verses")

    # format the data in CSV format
    formatted_data = "Verse No.,Text\n"
    for i in range(len(verses)):
        formatted_data = formatted_data + f"{verse_ids[i]},{' '.join(verses[i])}\n"

    # Store the scraped lines into a csv file
    p = Path("./out/scraper_results.csv")
    p.write_text(formatted_data)

    print("Process Complete!")
    print(f"number of hyperlinks found in the document: {num_links}")
    print(f"number of lemmata: {num_lemmata}")
    print(f"number of scribal variants: {num_variances}")