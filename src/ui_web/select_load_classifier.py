import logging
import re
from collections.abc import Iterable
from dataclasses import Field, dataclass, field, fields
from pathlib import Path
from typing import Literal
from venv import logger

from src.classifier.fname_utils import FnameExtraOpts, SavefileName, _sanitise
from src.classifier.result_utils import ResultStats
from src.shared import label_data
from src.ui_web.load_book_probas import BookProbas, load_book_probas
from src.ui_web.load_clf_stats import JsonifiedSummaryDict, load_clf_stats
from src.ui_web.load_predictions import BookVerses, load_preds

logger = logging.getLogger(__name__)

@dataclass
class ClassifierConfig:
    """Dataclass of configurations for the classifier."""

    name: str
    origin: Literal["CAL", "ETCBC"]
    # n-gram options
    is_n_gram: bool = False
    n: int = 0
    is_bow: bool = False
    is_char_level: bool = False
    # knn options
    k: int = 0
    weights: str = ""
    p: int = 0
    # extra options
    extra_opts: list[FnameExtraOpts] = field(default_factory=list)

    def fields(self) -> Iterable[Field]:
        """Returns the specs of fields."""
        return fields(self)

    def get_keys(self, attr_type: type | None = None) -> list[str]:
        """Returns the names of fields.

        Args:
            attr_type: type of the field to filter by. If ``None``, returns
                the names of all fields.
        """
        if attr_type is None:
            return [f.name for f in self.fields()]
        # else
        return [f.name for f in self.fields() if f.type == attr_type]


def configure_fname_opts(
    save_fname: SavefileName,
    configs: ClassifierConfig,
) -> SavefileName:
    """Configure the filename options for the classifier."""
    save_fname.set_ngram_opts(
        n=configs.n,
        is_n_gram=configs.is_n_gram,
        is_bow=configs.is_bow,
        is_char_level=configs.is_char_level,
    )
    save_fname.set_knn_opts(
        k=configs.k,
        weights=configs.weights,  # type: ignore[reportArgumentType]
        p=configs.p,
    )
    if configs.extra_opts is not None and len(configs.extra_opts) > 0:
        save_fname.add_extra_opts(configs.extra_opts)
    return save_fname


def parse_fname(fname: str) -> SavefileName:
    """Parse a file name and extract the classifier data file options.

    Raises:
        ValueError: if the file name is empty or does not contain a
            file extension.
    """
    fname = fname.strip()
    if not fname:
        msg = "Empty file name"
        raise ValueError(msg)
    # get the file extension
    fname_sections = fname.split(".")
    if len(fname_sections) < 2:
        msg = f"Missing file extension in: {fname}"
        raise ValueError(msg)
    fname_stem = _sanitise(fname_sections[0])
    f_ext = fname_sections[1]
    # split the filename into parts
    parts = fname_stem.split("_")
    # check if the file is for production data
    msg = f"Parsing file name from: {parts}"
    logger.debug(msg)
    is_prod = False
    if parts[0].upper() == "PRODUCTION":
        is_prod = True
        # remove the first word and continue parsing
        parts = parts[1:]
    # extract the origin and classifier alias
    origin = parts[0].upper()
    if origin not in {"CAL", "ETCBC"}:
        msg = f"Unknown data origin: {origin}"
        raise ValueError(msg)
    classifier_alias = parts[1]
    # initialise the save file name
    save_fname = SavefileName(origin, classifier_alias, f_ext)  # type: ignore[reportArgumentType]
    # check if the file is for a n-gram classifier
    ngram_fname = re.search(r"(char|word)_\dgram", fname)
    if ngram_fname is not None:
        # find the n-gram options
        ngram_opts = ngram_fname.group().split("_")
        is_char_level = (ngram_opts[0] == "char")
        n = ngram_opts[1].replace("gram", "")
        is_bow = re.search(r"_bow", fname) is not None
        # set the n-gram options
        save_fname.set_ngram_opts(
            n=int(n),
            is_n_gram=True,
            is_char_level=is_char_level,
            is_bow=is_bow,
        )
    fname_match = re.search(r"\dknn", fname)
    if fname_match:
        # find the knn options
        k = int(fname_match.group().replace("knn", ""))
        fname_weights = re.search(r"uniform|distance", fname)
        weights = fname_weights.group() if fname_weights else "uniform"
        fname_p = re.search(r"p\d", fname)
        p = int(fname_p.group().replace("p", "")) if fname_p else 2
        # set the knn options
        save_fname.set_knn_opts(
            k=int(k),
            weights=weights,  # type: ignore[reportArgumentType]
            p=p,
        )
    # check if it contains scope name
    for scope in label_data.LabelToVal:
        if scope.lower() in parts:
            save_fname.set_scope(scope)
            break
    # check if it is a special file
    if "mislabels" in parts:
        save_fname.mark_special_file(is_mislabel=True)
    elif "total" in parts and parts[(parts.index("total") + 1)] == "proba":
        save_fname.mark_special_file(is_prod=is_prod, is_total_proba=True)
    elif (
        "classifier" in parts
        and parts[(parts.index("classifier") + 1)] == "stats"
    ):
        save_fname.mark_special_file(is_prod=is_prod, is_clf_summary=True)
    elif is_prod:
        save_fname.mark_special_file(is_prod=True)
    # check for extra options
    extra_opts = []
    for opt in FnameExtraOpts:
        if opt.value in fname:
            extra_opts.append(opt)
    # set the extra options
    if len(extra_opts) > 0:
        save_fname.add_extra_opts(extra_opts)
    return save_fname


# TODO (mid-low priority): find some way to generate diagrams based on the
# results ... maybe using scikit-learn's \*Display classes with
# ``.from_predictions()`` methods?

# TODO (essential): create a file index to enable data retrieval for a
# particular classifier's results by matching the classifier configurations


class ResultFilesIndex:
    """A class to represent the index of classifier result files."""

    def __init__(self, load_dir: str | Path = "./src/classifier/out/"):
        self.load_dir: Path = Path(load_dir)
        self.files: list[SavefileName] = []
        self.file_paths: list[Path] = []
        # index the files in the directory
        for file in self.load_dir.iterdir():
            if file.is_file() and file.suffix in {".csv", ".json"}:
                # if the file is a data file
                # parse the file name(s) and append to the index
                save_fname = parse_fname(file.name)
                self.files.append(save_fname)
                self.file_paths.append(file)
                print(f"Indexed File: {file.name} -> {save_fname.get_fname()}")

    def __len__(self) -> int:
        """Returns the number of files in the index."""
        return len(self.files)

    def match_files_by_config(
        self, config: ClassifierConfig
    ) -> list[SavefileName]:
        """Match the file(s) in the index by the classifier configuration.

        Returns:
            a skeletal :class:`src.classifier.fname_utils.SavefileName` instance
            configured for a particular classifier.
        """
        # initialise the SavefileName object to match the files with
        sf = SavefileName(config.origin, config.name)
        configure_fname_opts(sf, config)

        # match the files in the index
        matched_fnames: list[SavefileName] = []
        for file in self.files:
            if sf.is_same_classifier(file):
                print(f"Matched File: {config} -> {sf.get_fname()}")
                matched_fnames.append(file)
        return matched_fnames


# TODO (essential): define a class to represent the classifier results
# (a model for loading data to be controlled by the Flask view functions
# = contain data to be listed on one HTML page)


class ClassifierResultsModel:
    """A class to represent the classifier results.

    This functions as a Model of the Model-View-Controller (MVC) framework,
    although in this case we deal with raw text files instead of DBs.
    """

    def __init__(
        self,
        config: ClassifierConfig,
        result_index: ResultFilesIndex,
        load_dir: str | Path = "./src/classifier/out/",
    ) -> None:
        self.config: ClassifierConfig = config
        self.index: ResultFilesIndex = result_index
        self.load_dir: Path = Path(load_dir)
        # initialise empty attributes
        self._is_loaded: bool = False
        self.files: list[SavefileName] = []
        self.book_probas: BookProbas | None = None
        self.book_verses: list[BookVerses] = []
        self.results_summary: JsonifiedSummaryDict | None = None

    def update_index(self, file_index: ResultFilesIndex) -> None:
        """Update the index of files."""
        self.index = file_index
        # re-load the results
        self.load_results()

    def load_results(self) -> None:
        """A utility to load data into the model.

        Raises:
            ValueError: if there is no data file matching the specified
            classifier configurations in the file index.
        """
        save_files = self.index.match_files_by_config(self.config)
        if len(save_files) == 0:
            msg = f"No files found for classifier: {self.config}"
            raise ValueError(msg)
        msg = f"Filtering from: {[f.get_fname() for f in save_files]}"
        print(msg)
        for file in save_files:
            if file.is_mislabel:
                # skip the file(s) for inspecting mislabelled instances
                continue

            if file.is_total_proba:
                self.book_probas = load_book_probas(file, self.load_dir)
            elif file.is_clf_summary:
                # load ResultStats from the summary (``classifier_stats``) json
                # file(s)
                self.results_summary = load_clf_stats(file, self.load_dir)
            else:
                # if the file is a normal CSV listing verses and their probas
                self.book_verses.extend(load_preds(file, self.load_dir))
            self.files.append(file)
        logger.debug(self.book_verses)
        # set the loaded flag to True
        self._is_loaded = True

    def _verify_load(self) -> None:
        """Verify that the data is loaded into the model.

        Raises:
            ValueError: if the data is not yet loaded.
        """
        if not self._is_loaded:
            msg = "Data not loaded. Call `load_results()` first."
            raise ValueError(msg)

    def get_book_verses(self) -> list[BookVerses]:
        """Get the book probas."""
        if not self._is_loaded:
            self.load_results()
        return self.book_verses
