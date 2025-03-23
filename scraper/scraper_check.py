from pathlib import Path
import re

fp = Path("./scraper_log.txt")

with fp.open() as f:
    for line in f:
        if re.match("Book:\\s.+") is not None:
            reset_counters(counts)
        elif re.match("\\d\\d:\\d\\d") is not None:
            check_counts(counts)
            counts["ref"] += 1
        elif re.match
