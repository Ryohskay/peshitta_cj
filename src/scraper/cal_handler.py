"""Script and functions to fetch HTML data from CAL."""

from pathlib import Path
from urllib.parse import urlencode

import urllib3

base_url = "https://cal.huc.edu/"


def pool_init() -> urllib3.PoolManager:
    """Initialise a PoolManager instance and return the object."""
    return urllib3.PoolManager()


def normalise_cset(cset: str) -> str:
    """Normalise the cset value.

    cset: ["R", "Roman", "L", "Latin", "S", "Syriac"]
        Defines the writing system (alphabets) in which to display the content.
        Corresponds to the `cset` URL query parameter.
        "R", "L", "Latin" are for Latin Alphabets, "S", "Syriac" for Syriac scripts.
        Note that "S" option displays Estrangelo in OT but Serto in NT.
    """
    cset = cset.strip()
    if cset in {"S", "Syriac"}:
        return "S"
    if cset in {"R", "Roman", "L", "Latin"}:
        return "R"
    msg = (
        f"cset {cset} is invalid. It must be one of: "
        + "'R', 'Roman', 'L', or 'Latin' for Roman transliteration, "
        + "and 'S' or 'Syriac' for Syriac scripts."
    )
    raise ValueError(msg)


def make_url(
    no_parse: bool = False,
    page: str = "get_a_chapter.php",
    query_params: None | dict = None,
    file: str = "62001",
    sub: None | int = None,
    cset: str = "Latin",
) -> str:
    """Construct a URL to query the CAL database with given parameters.

    See normalise_cset for param: cset.

    no_parse: Boolean
        Defines if the parameter parsing is needed.
        If True, simply returns a string combining the base_url and page.

    page: str
        sub-directory section of the URL, to be appended after the base URL

    query_params: None or dict
        dictionary containing all query parameters for the URL

    file: "%5d" (string of five-digit numeral)
        Corresponds to `file` URL query parameter.
        Must be a string of five-digit numeral to avoid passing "00001" as 1.
        Defaults to OT Peshitta Genesis.

    sub: None or int
        Corresponds to `sub` URL query parameter.
        The URL parameter must be in the form of "01", "02", ... etc. to work correctly.
        Defaults to None, in which case the whole book will be fetched.

    ERRORS:
        uses urllib.parse.urlencode() without try/catch.
    """
    if no_parse:
        return base_url + page

    if query_params is not None:
        # if query_params are explicitly given, return a URL string with the params
        return base_url + page + "?" + urlencode(query_params)

    # If no_parse is False and query_params are not given
    # Construct a dictionary of parameters
    params = {"file": file, "cset": normalise_cset(cset)}

    if sub is not None:
        # Add "sub" query param
        params.update({"sub": f"{sub:02}"})

    return base_url + page + "?" + urlencode(params)


def get_a_chapter(
    pool: urllib3.PoolManager,
    book_id: str = "62001",
    section: None | int = None,
) -> str:
    """Query the https://cal.huc.edu/get_a_chapter.php endpoint with given parameters.

    If no section is specified, the whole book will be fetched.
    Returns the responded html after decoding in UTF-8.

    See make_url for params: book_id, section.

    pool: urllib3.PoolManager
        Connection pool for urllib3.

    ERRORS:
        uses urllib3.PoolManager.request() without try/catch.
    """
    res = pool.request("GET", make_url(file=book_id, sub=section))
    return res.data.decode(
        "utf-8"
    )  # return decoded text from response text html


def get_a_syriac_chapter(
    pool: urllib3.PoolManager,
    book_id: str = "62001",
    section: None | int = None,
    needs_est: bool = False,
) -> str:
    """Query the https://cal.huc.edu/get_a_chapter.php endpoint with given parameters.

    If no section is specified, the whole book will be fetched.
    Returns the responded html after decoding in UTF-8.

    See normalise_cset for param: display_in.
    See make_url for params: book_id,  and section.

    pool: urllib3.PoolManager
        Connection pool for urllib3.

    needs_est: bool
        True if the book/chapter requires a conversion into Estrangelo.
        If this is set to True, then query get_a_chapterEST.php endpoint.

    ERRORS:
        uses urllib3.PoolManager.request() without try/catch.
    """
    if needs_est:
        res = pool.request(
            "GET",
            make_url(
                page="get_a_chapterEST.php",
                file=book_id,
                sub=section,
                cset="Syriac",
            ),
        )
    else:
        res = pool.request(
            "GET",
            make_url(
                file=book_id,
                sub=section,
                cset="Syriac",
            ),
        )
    return res.data.decode(
        "utf-8"
    )  # return decoded text from response text html


def get_and_save(
    fpath: str | Path,
    pool_mgr: urllib3.PoolManager,
    book_id: str,
    section: int,
    display_in: str = "Latin",
    allow_overwrite: bool = False,
    needs_est: bool = False,
):
    """Query using get_a_chapter, save the resulting HTML in a file.

    See normalise_cset for param: display_in
    See get_a_syriac_chapter for param: needs_est
    See make_url function for params: pool_mgr, book_id, section,

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
    if normalise_cset(display_in) == "S":
        qr = get_a_syriac_chapter(pool_mgr, book_id, section, needs_est)
    else:
        qr = get_a_chapter(pool_mgr, book_id, section)
    dest = fpath if isinstance(fpath, Path) else Path(fpath)

    if dest.exists() and not dest.is_file():
        msg = f"The specified path {fpath} exists, and it is not a file path."
        raise ValueError(msg)

    # Avoid overwriting files unless explicitly allowed
    if dest.exists() and not allow_overwrite:
        msg = f"Overwrite not allowed: The file {fpath} exists, and allow_overwrite parameter is False!"
        raise FileExistsError(msg)
    if dest.exists():
        print(f"Overwriting: {fpath}")

    dest.write_text(qr)


def follow_link(pool: urllib3.PoolManager, link_url: str) -> str:
    """Follow the url to the getlex.php result page and get lemma(ta).

    pool: urllib3.PoolManager
        Connection pool for urllib3.

    link_url: str
        relative url link to the lemma(ta) page.
    """
    res = pool.request("GET", make_url(no_parse=True, page=link_url))
    return res.data.decode(
        "utf-8"
    )  # return decoded text from response text html


def get_verse_url(url: str, cset: str = "Latin"):
    """Derive the url to a verse page from given url to the lexicon entry.

    This function receives a url to a lexicon entry and extracts
    the coord query parameter in the url.
    """
    verse_ref = ""
    # print(f"parsing {url}")
    parsed_url = urllib3.util.parse_url(url)
    query_params = parsed_url.query.split("&")
    for param in query_params:
        if "coord" in param:
            verse_ref = param.split("=")[1]
    v_url = make_url(file=verse_ref, cset=cset)
    print(f"v_url: {v_url}")
    return v_url


if __name__ == "__main__":
    http = urllib3.PoolManager()
    get_and_save("./out/example.html", http)
    print("Process Complete!")
