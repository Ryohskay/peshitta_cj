"""The main scraping script to get Syriac texts from CAL."""

import json
from bs4 import BeautifulSoup, Tag
from cal_handler import (pool_init, get_a_chapter, get_a_syriac_chapter,
                         follow_link, get_verse_url, normalise_cset)
from pathlib import Path
import re
import time
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from cal_handler import follow_link, get_a_chapter, get_verse_url, pool_init


def count_char(target: str, source: str) -> int:
    """Count the number of target characters."""
    num = 0
    for letter in source:
        if letter == target:
            num = num + 1
    return num


def get_lemma_line(markup_tag: Tag) -> str | None:
    """Get the line in the lexicon entry where lemma is listed.

    markup_tag: Tag
        result(s) returned by bs4's find() or find_all(),
        an excerpt of a HTML document. 
    """
    stripped = markup_tag.text.strip()
    # remove trailing spaces and self-standing periods.
    # idk why but CAL places periods outside of any html tags,
    # which the HTML parser considers as a raw string under
    # the body tag.
    in_tag_txt = re.sub("^\\.$", "", stripped)
    if markup_tag.name is None and in_tag_txt != "":
        # if the content of body tag is not within any tag
        # but a raw string directly under body tag
        # (this is how CAL lists lemmata in lexcicon entries)
        return in_tag_txt
    return None


def is_lemma(lex: str) -> bool:
    """Verifies if given string is a valid lemma."""
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


def extract_lemma_annot(tag_txt: str) -> tuple[str | None, str | None]:
    """Extract the lemma and annotation from given document excerpt.
    
    Note that the lemma annotations may not provide POS at all,
    e.g. https://cal.huc.edu/getlex.php?coord=620010102&word=9

    tag_txt: str
        string returned by get_lemma_line()
    """
    if is_lemma(tag_txt):
        # get the lemma and remove POS description etc.
        split_line = tag_txt.split(" ")
        lemma = split_line[0]
        # remove numbering appended to lemma, like "???#2"
        # and replace "@" in compound words with a space
        lemma = re.sub("#\\d", "", lemma).replace("@", " ")
        # record the annotations if they exist
        if len(split_line) > 1:
            annot = " ".join(split_line[1:]) # lemma annotations like POS, morphology
        else:
            print(f"INFO: Annotation not given for lemma: {lemma}")
            annot = None
        return (lemma, annot)
    return (None, tag_txt)


def is_lex(url: str) -> bool:
    """Check if the url leads to a lexicon entry."""
    if "getlex" in url:
        return True
    return False


def is_empty_verse(words: list) -> bool:
    """Check if the verse is all None."""
    for w in words:
        if w is not None:
            return False
    return True


if __name__ == "__main__":

    OT = False
    NT = True

    target_books = [
            ("Matthew", "62040", NT),
            ("Mark", "62041", NT),
            ("Luke", "62042", NT),
            ("John", "62043", NT),
            ("Genesis", "62001", OT),
            ("Exodus", "62002", OT),
            ("Acts", "62044", NT),
            ("Deuteronomy", "62005", OT),
            ("Joshua", "62006", OT),
            ("Judges", "62007", OT),
            ("1_Samuel", "62008", OT),
            ("2_Samuel", "62009", OT),
            ("1_Kings", "62010", OT),
            ("2_Kings", "62011", OT),
            ("Ruth", "62030", OT),
            ("Esther", "62034", OT),
            ("Ezra", "62036", OT),
            ("Nehemiah", "62037", OT),
            ("1_Chronicles", "62038", OT),
            ("2_Chronicles", "62039", OT),
            ("1_Maccabees", "62078", OT),
            ]

    # book_idx = "62006"

    cset = "Syriac"

    print(time.asctime())

    http = pool_init()
    for book in target_books:
        result = None
        print(f"Book: {book[0]}")
        if normalise_cset(cset) == "S":
            result = get_a_syriac_chapter(http, book_id=book[1], needs_est=book[2])
        else:
            result = get_a_chapter(http, book_id=book[1], display_in=cset)
    # for i in range(15,16):
        # book_idx = target_books[0]
        # print(f"Chapter: {i}")
        # result = get_a_chapter(http, book_id=book_idx, section=i)

        # Use lxml parser to correctly handle raw texts in body tag
        soup = BeautifulSoup(result, "lxml")

        # book-level vars
        # lists to store results
        verses = []
        raw_verses = [] # aggregate list of inflected texts from all verses
        verse_ref_nums = [] # the reference numbers for each verse
        empty_verses = [] # verse reference to verses where no lemma could be
                          # retrieved
        error_lines = [] # lines where DB error is suspected
        xx_lines = [] # lines where the reference is XX
        verse_urls = [] # list of verse urls
        verse_annots = [] # aggregate list of verse annotations

        # book-level counters
        num_slashes = 0
        num_lemmata = 0
        num_no_lemma = 0
        num_links = 0

        # verse-level vars
        word_lis = []  # lemmatised words of each verse as a list of lemmata
        raw_words = [] # inflected form of words in each verse, as a list
        annot_lis = [] # annotations of the lemma, as a list of str
        lex_url = ""
        verse_url = ""

        # verse-level flags
        is_verse_line = False

        for table_data in soup.find_all("td"):
            if "valign" in table_data.attrs.keys() and table_data["valign"] == "top":
                stripped = str(table_data.text).strip()
                # when the scraper reaches a new row on the table.
                # add the verse identifier (e.g. "01:01")
                verse_ref = re.match("\\d\\d:\\d\\d", stripped)
                if verse_ref is not None:
                    vid = verse_ref.group()
                    if normalise_cset(cset) == "S":
                        vid = vid[::-1]  # reverse the ref, it's in bdo tag
                    print(vid)
                    v_refs = vid.split(":")
                    reference = (
                            f"{book[0]} Chapter {v_refs[0]}"
                            + f" Verse {v_refs[1]}"
                            )
                    verse_ref_nums.append(reference)
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
                for link in table_data.find_all("a"):

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
                        lex_soup = BeautifulSoup(lex_page, "lxml")
                        lex_body = lex_soup.body

                        for body_child in lex_body.children:
                            # for each DOM object directly under the body tag
                            lemma_line = get_lemma_line(body_child)
                            if lemma_line is not None:
                                res, annot = extract_lemma_annot(lemma_line)
                                word_lis.append(res)
                                annot_lis.append(annot)
                                if res is None:
                                    num_no_lemma += 1
                    else:
                        word_lis.append(None)
                        annot_lis.append(None)
                        num_no_lemma += 1

                # extract the hyperlink to verse from the lex_url
                if verse_url != "":
                    extracted = get_verse_url(verse_url, cset=cset)
                    verse_urls.append(extracted)


                # when all links in one table cell has been explored,
                # push the list of scraped lemmata to verses[]
                print(word_lis)
                num_lemmata = num_lemmata + len(word_lis)
                verses.append(word_lis)
                raw_verses.append(raw_words)
                verse_annots.append(annot_lis)
                if len(word_lis) == 0 or is_empty_verse(word_lis):
                    empty_verses.append(verse_ref_nums[-1])
                # reset the verse-level lists
                word_lis = []
                raw_words = []
                annot_lis = []

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

        # # format the data in a string of CSV format
        # formatted_data = (
        #         'Verse Ref. No.,Verse URL,Raw Text,'
        #         + 'Lemmatised Text,Lemma Annotations\n'
        #         )
        # print(verse_annots)
        # for i in range(len(verses)):
        #     indices = verse_ref_nums[i].split(':')
        #     raw_txt_verse = ' '.join(raw_verses[i])
        #     lemma_verse = '#/#'.join(verses[i])
        #     annot_verse = '#/#'.join(verse_annots[i])
        #     formatted_data = (
        #             formatted_data
        #             + f'"Chapter {indices[0]} verse {indices[1]}",'
        #             + '"{verse_urls[i]}","{raw_txt_verse}",'
        #             + '"{lemma_verse}","{annot_verse}"\n'
        #             )

        # Store the scraped lines into a csv file
        with Path(f"./out/scraper_results_{cset}_{book[0]}.json").open(mode="w") as fp:
            book_data = {
                    "book_title": book[0],
                    "verse_refs": verse_ref_nums,
                    "verse_urls": verse_urls,
                    "raw_text_verses": raw_verses,
                    "lemmatised_verses": verses,
                    "lemma_annotations": verse_annots
                    }
            json.dump(book_data, fp)

        print(f"Done scraping for book {book[0]}")
        print(time.asctime())
        print("--RESULTS--")
        print(f"number of hyperlinks found in the document: {num_links}")
        print(f"number of lemmata: {num_lemmata}")
        print(f"number of empty lemmata: {num_no_lemma}")
        print(f"number of real lemmata: {num_lemmata - num_no_lemma}")
        print(f"number of scribal variants: {num_variances}")
        print(f"number of verses: {len(verses)}")
        print(f"number of empty verses: {len(empty_verses)}")
        print(empty_verses)
        print(f"number of verses whose location is uncertain: {len(xx_lines)}")
        print(xx_lines)
