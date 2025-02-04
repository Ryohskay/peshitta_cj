"""The main scraping script to get Syriac texts from CAL."""

from cal_handler import pool_init, get_and_save

http = pool_init()
get_and_save("./out/example.html", http, allow_overwrite=True)
print("Process Complete!")