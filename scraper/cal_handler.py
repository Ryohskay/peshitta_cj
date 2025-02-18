"""Script and functions to fetch HTML data from CAL."""

from urllib.parse import urlencode
from pathlib import Path
import urllib3

base_url = "https://cal.huc.edu/"


def pool_init() -> urllib3.PoolManager:
    """Initialise a PoolManager instance and return the object."""
    return urllib3.PoolManager()

def get_a_chapter(
        pool: urllib3.PoolManager,
        book_id: str="62001", 
        section: int=None, 
        display_in: str="Latin") -> str:
    """Query the https://cal.huc.edu/get_a_chapter.php endpoint with given parameters.

    If no section is specified, the whole book will be fetched.
    Returns the responded html after decoding in UTF-8.

    pool: urllib3.PoolManager
        Connection pool for urllib3.

    book_id: "%5d" (string of five-digit numeral)
        Corresponds to `file` URL query parameter.
        Must be a string of five-digit numeral to avoid passing "00001" as 1.
        Defaults to OT Peshitta Genesis.

    section: int
        Corresponds to `sub` URL query parameter.
        The URL parameter must be in the form of "01", "02", ... etc. to work correctly.
        Defaults to None, in which case the whole book will be fetched.

    display_in: ["R", "L", "Latin", "S", "Syriac"]
        Defines the writing system (alphabets) in which to display the content.
        Corresponds to the `cset` URL query parameter.
        "R", "L", "Latin" are for Latin Alphabets, "S", "Syriac" for Syriac scripts.
        Note that "S" option displays Estrangelo in OT but Serto in NT.

    ERRORS:
        uses urllib.parse.urlencode() without try/catch.
        uses urllib3.PoolManager.request() without try/catch.
    """

    params = {
        "file": book_id,
        "cset": display_in
    }

    if section is not None:
        # Add "sub" query param
        params.update({"sub": f"{section:02}"})

    query_url = base_url + "get_a_chapter.php?" + urlencode(params)
    res = pool.request("GET", query_url)
    return res.data.decode("utf-8")  # return decoded text from response text html


def get_and_save(
        fpath: str | Path,
        pool_mgr: urllib3.PoolManager,
        book_id: str="62001", 
        section: str=None, 
        display_in: str="Latin",
        allow_overwrite: bool=False
        ):
    """Query using get_a_chapter, save the resulting HTML in a file.
    
    See get_a_chapter function for detailed requirements of:
        pool_mgr, book_id, section, display_in

    If no section is specified, the whole book will be fetched.

    fpath: str | pathlib.Path
        The name of the file to save the data fetched from CAL.
        It's passed to pathlib, so it can handle both POSIX and
        MS Windows paths.

    ERRORS:
        raises ValueError if the path in fpath param is not a file.
        raises FileExistsError if the save file specified in fpath param already exists.
        uses pathlib.Path.write_text() without try/catch.
    """
    qr = get_a_chapter(pool_mgr, book_id, section, display_in)
    if isinstance(fpath, Path):
        dest = fpath
    else:
        dest = Path(fpath)
    
    if dest.exists() and not dest.is_file():
        raise ValueError("The specified path {} exists, and it is not a file path.".format(fpath))

    # Avoid overwriting files unless explicitly allowed
    if dest.exists() and not allow_overwrite:
        raise FileExistsError("Overwrite not allowed: The file {} exists, and allow_overwrite parameter is False!".format(fpath))
    elif dest.exists():
        print("Overwriting: {}".format(fpath))

    dest.write_text(qr)


def follow_link(pool: urllib3.PoolManager, link_url: str)  -> str:
    """Follow the url to the getlex.php result page and get lemma(ta).
    
    pool: urllib3.PoolManager
        Connection pool for urllib3.
    
    link_url: str
        relative url link to the lemma(ta) page.
    """
    lex_url = base_url + link_url
    res = pool.request("GET", lex_url)
    return res.data.decode("utf-8")  # return decoded text from response text html


def get_verse_url(lex_url: str):
    """Derive the url to a verse page from given url to the lexicon entry."""
    


if __name__ == "__main__":
    http = urllib3.PoolManager()
    get_and_save("./out/example.html", http)
    print("Process Complete!")