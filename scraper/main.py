"""The main scraping script to get Syriac texts from CAL."""

from bs4 import BeautifulSoup, Tag
from cal_handler import pool_init, get_a_chapter, follow_link, get_verse_url
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
    anomalous_words = [
            "part of previous word",
            "non-Aramaic or fragmentary",
            "there is no data" # https://cal.huc.edu/getlex.php?coord=620411620&word=15
            ]
    for keywords in anomalous_words:
        if keywords in lex:
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
        # but a raw string (this is how CAL lists lemmata)
        stripped = markup_tag.text.strip()
        # remove trailing periods lying outside the tags
        res = re.sub("^\\.$", "", stripped)
        if res != "" and is_lemma(res):
            # get the lemma and remove POS description etc.
            res = res.split(' ')[0]
            # remove numbering appended to lemma, like "???#2"
            # and replace "@" in compound words with a space
            return re.sub("#\\d", "", res).replace("@", " ")


def is_lex(url: str) -> bool:
    if "getlex" in url:
        return True
    return False


if __name__ == "__main__":

    target_books = [
            # ("Matthew", "62040"),
            # ("Mark", "62041"),
            # ("Luke", "62042"),
            # ("John", "62043"),
            # ("Genesis", "62001"),
            # ("Exodus", "62002"),
            # ("Acts", "62044"),
            # ("Deuteronomy", "62005"),
            # ("Joshua", "62006"),
            # ("Judges", "62007"),
            # ("1_Samuel", "62008"),
            # ("2_Samuel", "62009"),
            # ("1_Kings", "62010"),
            # ("2_Kings", "62011"),
            # ("Ruth", "62030"),
            # ("Esther", "62034"),
            # ("Ezra", "62036"),
            # ("Nehemiah", "62037"),
            # ("1_Chronicles", "62038"),
            # ("2_Chronicles", "62039"),
            ("1_Maccabees", "62078"),
            ]

    #book_idx = "62006"

    http = pool_init()
    for book in target_books:
        print(f"Book: {book[0]}")
        result = get_a_chapter(http, book_id=book[1])
    # for i in range(15,16):
        # book_idx = target_books[0]
        # print(f"Chapter: {i}")
        # result = get_a_chapter(http, book_id=book_idx, section=i)

        # Use lxml parser to correctly handle raw texts in body tag
        soup = BeautifulSoup(result, 'lxml')

        # lists to store results
        verses = []
        raw_verses = [] # aggregate list of inflected texts from all verses
        raw_words = [] # inflected form of words in each verse, as a list
        word_lis = []  # lemmatised words of each verse as a list of lemmata
        verse_ref_nums = [] # the reference numbers for each verse
        empty_verses = [] # verse reference to verses where no lemma could be
                          # retrieved
        error_lines = [] # lines where DB error is suspected
        xx_lines = [] # lines where the reference is XX
        verse_urls = []

        lex_url = ""
        verse_url = ""

        # flags
        is_verse_line = False

        # counters
        num_slashes = 0
        num_lemmata = 0
        num_no_lemma = 0
        num_links = 0

        for table_data in soup.find_all('td'):
            if "valign" in table_data.attrs.keys() and table_data["valign"] == "top":
                # when the scraper reaches a new row on the table.
                # add the verse identifier (e.g. "01:01")
                verse_ref = re.match("\\d\\d:\\d\\d", table_data.text)
                if verse_ref is not None:
                    vid = verse_ref.group()
                    verse_ref_nums.append(vid)
                    print(vid)
                    is_verse_line = True
                elif "d.:ne" in table_data.text:
                    # probably some error in the database, skip the line
                    # e.g. Matthew 19:03 from
                    # https://cal.huc.edu/get_a_chapter.php?file=62040&cset=Latin
                    error_lines.append(f"{verse_ref_nums[-1]} +1")
                    is_verse_line = False
                    print("INFO: Found an errorneous line after verse " +
                          f"{verse_ref_nums[-1]}, skipping")
                elif "xx" in table_data.text.lower():
                    # Handle cases like 2 Samuel 24:XX
                    xx_lines.append(f"{verse_ref_nums[-1]} +1")
                    vid = table_data.text.strip()
                    verse_ref_nums.append(vid)
                    print("INFO: Found a line with reference no." +
                          f" {table_data.text}" +
                          f" after verse {verse_ref_nums[-1]}")
                    is_verse_line = True
                else:
                    print("INFO: Found td cell with valign attr that is not a verse reference")

            elif len(table_data.text) > 0 and is_verse_line:  # If it's the cell containing verse
                # go through all links in the table data cell
                for link in table_data.find_all('a'):

                    lex_url = link["href"]

                    if is_lex(lex_url):
                        # if it's linked to getlex.php file, it's a word
                        # remember the url to the lexeme
                        verse_url = lex_url
                        # count the number of slash
                        num_slashes = num_slashes + count_char("/", link.text)
                        # get the inflected, non-lemmatised word
                        raw_words.append(link.text.strip())
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
                    else:
                        word_lis.append(None)
                        num_no_lemma += 1

                # extract the hyperlink to verse from the lex_url
                if verse_url != "":
                    verse_url = get_verse_url(lex_url)
                    verse_urls.append(verse_url)

                # when all links in one table cell has been explored,
                # push the list of scraped lemmata to verses[]
                #if len(word_lis) > 0:
                print(word_lis)
                num_lemmata = num_lemmata + len(word_lis)
                raw_verses.append(raw_words)
                verses.append(word_lis)
                if len(word_lis) == 0:
                    empty_verses.append(verse_ref_nums[-1])
                # reset the lists
                word_lis = []
                raw_words = []

        # the number of scribal variances is number of slashes in raw text divided by 2
        num_variances = num_slashes / 2

        if len(verses) != len(verse_ref_nums):
            print("INFO: the number of verses and verse_ref_nums do not match: ")
            print(f"verse_ref_nums: {len(verse_ref_nums)} but" +
                             f" verses: {len(verses)}," +
                             f" raw_verses: {len(raw_verses)}" +
                             f" empty_verses: {len(empty_verses)}"
                             )
            print(f"empty verses: {empty_verses}")

        # format the data in a string of CSV format
        formatted_data = "Verse Ref. No.,Verse URL,Raw Text,Lemmatised Text\n"
        for i in range(len(verses)):
            indices = verse_ref_nums[i].split(":")
            raw_txt_verse = ' '.join(raw_verses[i])
            lemma_verse = ' '.join(verses[i])
            formatted_data = formatted_data + 
            f'"Chapter {indices[0]} verse {indices[1]}","{verse_urls[i]}","{raw_txt_verse}","{lemma_verse}"\n'

        # Store the scraped lines into a csv file
        p = Path(f"./out/scraper_results_{book[0]}.csv")
        p.write_text(formatted_data)

        print(f"Done scraping for book {book[0]}")
        print("--RESULTS--")
        print(f"number of hyperlinks found in the document: {num_links}")
        print(f"number of lemmata: {num_lemmata}")
        print(f"number of scribal variants: {num_variances}")
        print(f"number of verses: {len(verses)}")
        print(f"number of empty verses: {len(empty_verses)}")
        print(empty_verses)
        print(f"number of verses whose location is uncertain: {len(xx_lines)}")
        print(xx_lines)
