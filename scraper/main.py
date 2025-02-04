"""The main scraping script to get Syriac texts from CAL."""

from bs4 import BeautifulSoup
from cal_handler import pool_init, get_and_save
from pathlib import Path

http = pool_init()
save_f = Path("./out/example.html")
get_and_save(save_f, http, allow_overwrite=True)

soup = BeautifulSoup(save_f.open(), 'html.parser')

for link in soup.find_all('a'):
    print(link)

print("Process Complete!")