from bs4 import BeautifulSoup, Tag
from cal_handler import pool_init, get_a_chapter, follow_link, get_verse_url
from pathlib import Path
from main import is_lex, count_char, find_lemma
import re

if __name__ == "__main__":
    book_idx = "62006"

    http = pool_init()
    #for book_idx in target_books:
    i = 15
    #book_idx = target_books[0]
    #print(f"Chapter: {i}")
    #result = get_a_chapter(http, book_id=book_idx)
    result = get_a_chapter(http, book_id=book_idx, section=i)

    # Use lxml parser to correctly handle raw texts in body tag
    soup = BeautifulSoup(result, 'lxml')

    # lists to store results
    verses = []
    raw_verses = [] # aggregate list of inflected texts from all verses
    raw_words = [] # inflected form of words in each verse, as a list
    word_lis = []  # lemmatised words of each verse as a list of lemmata
    verse_ref_nums = [] # the reference numbers for each verse
    verse_urls = []

    lex_url = ""
    verse_url = ""

    # counters
    num_slashes = 0
    num_lemmata = 0
    num_links = 0

    print(soup.find_all("td"))
    table_data = soup.find("td", string=re.compile("33")).find_next_sibling("td")
    if len(table_data.text) <= 0:
        print(f"table_data: {table_data.text}")
    elif len(table_data.text) > 0:  # If it's the cell containing verse
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

        # extract the hyperlink to verse from the lex_url
        if verse_url != "":
            verse_url = get_verse_url(lex_url)
            verse_urls.append(verse_url)

        # when all links in one table cell has been explored,
        # push the list of scraped lemmata to verses[]
        if len(word_lis) > 0:
            print(word_lis)
            num_lemmata = num_lemmata + len(word_lis)
            verses.append(word_lis)
            raw_verses.append(raw_words)
            # reset the lists
            word_lis = []
            raw_words = []

            # extract the hyperlink to verse from the lex_url
            # if verse_url != "":
            #     verse_url = get_verse_url(lex_url)
            #     verse_urls.append(verse_url)

        # the number of scribal variances is number of slashes divided by 2
        num_variances = num_slashes / 2

        if len(verses) != len(verse_ref_nums):
            print("Something is wrong with the number of verses vs verse_ref_nums")
            raise ValueError(f"verse_ref_nums: {len(verse_ref_nums)} but" +
                             f" verses: {len(verses)}, raw_verses: {len(raw_verses)}")

        # format the data in a string of CSV format
        formatted_data = "Verse Ref. No.,Verse URL,Raw Text,Lemmatised Text\n"
        for i in range(len(verses)):
            indices = verse_ref_nums[i].split(":")
            raw_txt_verse = ' '.join(raw_verses[i])
            lemma_verse = ' '.join(verses[i])
            formatted_data = formatted_data + f'"Chapter {indices[0]} verse {indices[1]}",{verse_urls[i]},"{raw_txt_verse}","{lemma_verse}"\n'

        # Store the scraped lines into a csv file
        # p = Path(f"./out/scraper_results_{book_idx}.csv")
        # p.write_text(formatted_data)

        print(f"Done scraping for book {book_idx}")
        print(f"number of hyperlinks found in the document: {num_links}")
        print(f"number of lemmata: {num_lemmata}")
        print(f"number of scribal variants: {num_variances}")
