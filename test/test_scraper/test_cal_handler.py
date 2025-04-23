from collections.abc import Callable
import pytest
from pathlib import Path
from unittest.mock import MagicMock
import urllib3
from src.scraper.cal_handler import (
    pool_init,
    normalise_cset,
    make_url,
    get_a_chapter,
    get_a_syriac_chapter,
    get_and_save,
    follow_link,
    get_verse_url,
)


@pytest.fixture
def mock_pool_manager():
    """Fixture to provide a mocked urllib3.PoolManager."""
    return MagicMock(spec=urllib3.PoolManager)


def test_pool_init():
    """Test the pool_init function."""
    pool = pool_init()
    assert isinstance(pool, urllib3.PoolManager)


def test_normalise_cset():
    """Test the normalise_cset function."""
    assert normalise_cset("Latin") == "R"
    assert normalise_cset("Syriac") == "S"
    assert normalise_cset("R") == "R"
    assert normalise_cset("S") == "S"

    with pytest.raises(ValueError, match="cset Arabic is invalid"):
        normalise_cset("Arabic")


def test_make_url():
    """Test the make_url function."""
    url = make_url(file="62001", sub=1, cset="Latin")
    assert url == "https://cal.huc.edu/get_a_chapter.php?file=62001&cset=R&sub=01"

    url_no_parse = make_url(no_parse=True, page="test_page.php")
    assert url_no_parse == "https://cal.huc.edu/test_page.php"

    with pytest.raises(ValueError, match="cset Cyrillic is invalid"):
        make_url(file="62001", sub=1, cset="Cyrillic")


def test_get_a_chapter(mock_pool_manager):
    """Test the get_a_chapter function."""
    mock_pool_manager.request.return_value.data = b"<html>Chapter Content</html>"
    result = get_a_chapter(mock_pool_manager, book_id="62001", section=1)
    assert result == "<html>Chapter Content</html>"
    mock_pool_manager.request.assert_called_once_with(
        "GET", "https://cal.huc.edu/get_a_chapter.php?file=62001&cset=R&sub=01"
    )


def test_get_a_syriac_chapter(mock_pool_manager):
    """Test the get_a_syriac_chapter function."""
    mock_pool_manager.request.return_value.data = b"<html>Syriac Chapter Content</html>"
    result = get_a_syriac_chapter(mock_pool_manager, book_id="62001", section=1)
    assert result == "<html>Syriac Chapter Content</html>"
    mock_pool_manager.request.assert_called_once_with(
        "GET", "https://cal.huc.edu/get_a_chapter.php?file=62001&cset=S&sub=01"
    )


@pytest.fixture
def mock_normalise_cset() -> Callable[[str], str]:
    def _normalise(cset):
        return "R" if cset == "Latin" else "S"
    return _normalise

def test_get_and_save(tmp_path, mock_pool_manager, monkeypatch, mock_normalise_cset):
    """Test the get_and_save function."""
    mock_pool_manager.request.return_value.data = b"<html>Chapter Content</html>"

    # Mock the normalise_cset function
    monkeypatch.setattr("src.scraper.cal_handler.normalise_cset", mock_normalise_cset)

    file_path = tmp_path / "chapter.html"
    get_and_save(
        fpath=file_path,
        pool_mgr=mock_pool_manager,
        book_id="62001",
        section=1,
        display_in="Latin",
        allow_overwrite=False,
    )

    assert file_path.exists()
    assert file_path.read_text() == "<html>Chapter Content</html>"


def test_get_and_save_overwrite(tmp_path, mock_pool_manager, monkeypatch, mock_normalise_cset):
    """Test the get_and_save function with overwrite enabled."""
    mock_pool_manager.request.return_value.data = b"<html>Updated Content</html>"

    # Mock the normalise_cset function
    monkeypatch.setattr("src.scraper.cal_handler.normalise_cset", mock_normalise_cset)

    file_path = tmp_path / "chapter.html"
    file_path.write_text("<html>Old Content</html>")

    get_and_save(
        fpath=file_path,
        pool_mgr=mock_pool_manager,
        book_id="62001",
        section=1,
        display_in="Latin",
        allow_overwrite=True,
    )

    assert file_path.exists()
    assert file_path.read_text() == "<html>Updated Content</html>"


def test_follow_link(mock_pool_manager: MagicMock):
    """Test the follow_link function."""
    mock_pool_manager.request.return_value.data = b"<html>Lemma Content</html>"
    result = follow_link(mock_pool_manager, link_url="getlex.php?param=value")
    assert result == "<html>Lemma Content</html>"
    mock_pool_manager.request.assert_called_once_with(
        "GET", "https://cal.huc.edu/getlex.php?param=value"
    )


def test_get_verse_url(monkeypatch):
    """Test the get_verse_url function."""
    def mock_parse_url(url):
        return MagicMock(query="coord=62001")

    monkeypatch.setattr(
        "src.scraper.cal_handler.urllib3.util.parse_url",
        mock_parse_url,
    )
    result = get_verse_url("https://cal.huc.edu/getlex.php?coord=62001", cset="Latin")
    assert result == "https://cal.huc.edu/get_a_chapter.php?file=62001&cset=R"