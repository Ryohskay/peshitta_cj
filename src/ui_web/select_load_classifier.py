from src.classifier.fname_utils import SavefileName, FnameExtraOpts
from typing import Iterable, Literal, TypedDict
import re
from pathlib import Path
from src.ui_web.load_book_probas import BookProbas, load_book_probas
from src.ui_web.load_predictions import load_preds, BookVerses
from src.classifier.result_utils import ResultStats, Verse, ThresholdStats
from dataclasses import dataclass, fields, field, Field

@dataclass
class ClassifierConfig:
    """Configuration for the classifier."""
    name: str = ""
    origin: Literal["CAL", "ETCBC"] = "CAL"
    # n-gram options
    is_n_gram: bool = False
    n: int = 0
    is_bow: bool = False
    is_char_level: bool = False
    extra_opts: list[FnameExtraOpts] = field(default_factory=list)

    def fields(self) -> Iterable[Field]:
        """Returns the specs of fields."""
        return fields(self)

    def get_keys(self, attr_type: type | None=None) -> list[str]:
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
    save_fname.set_ngram_opts(n = configs.n,
                                is_n_gram = configs.is_n_gram,
                                is_bow = configs.is_bow,
                                is_char_level = configs.is_char_level)
    if configs.extra_opts is not None:
        save_fname.add_extra_opts(configs.extra_opts)
    return save_fname


def parse_fname(fname: str) -> SavefileName:
    """Parse a file name and extract the classifier data file options."""
    # split the filename into parts
    parts = fname.split("_")
    # check if the file is for production data
    is_prod = False
    if parts[0] == "PRODUCTION":
        is_prod = True
        # remove the first word and continue parsing
        parts = parts[1:]
    # extract the origin and classifier alias
    origin = parts[0]
    if origin not in {"CAL", "ETCBC"}:
        msg = f"Unknown data origin: {origin}"
        raise ValueError(msg)
    classifier_alias = parts[1]
    # initialise the save file name
    save_fname = SavefileName(origin, classifier_alias)  # type: ignore[reportArgumentType]
    # check if the file is for a n-gram classifier
    is_n_gram = (re.search(r"\dgram", fname) is not None)
    if is_n_gram:
        # find the n-gram options
        is_char_level = (parts[2] == "char")
        n = parts[3].replace("gram", "")
        is_bow = (parts[4] == "bow")
        # set the n-gram options
        save_fname.set_ngram_opts(n=int(n), is_n_gram=is_n_gram,
                                    is_char_level=is_char_level,
                                    is_bow=is_bow)
    # check if it is a special file
    if "_mislabels" in parts:
        save_fname.mark_special_file(is_prod=is_prod, is_mislabel=True)
    elif "_total_proba" in parts:
        save_fname.mark_special_file(is_prod=is_prod, is_total_proba=True)
    elif "_classifier_stats" in parts:
        save_fname.mark_special_file(is_prod=is_prod, is_clf_summary=True)
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

    def __init__(self,
                load_dir: str | Path = "./src/classifier/out/"):
        self.load_dir: Path = Path(load_dir)
        self.files: list[SavefileName] = []
        self.file_paths: list[Path] = []
        # index the files in the directory
        for file in self.load_dir.iterdir():
            if file.is_file():
                # parse the file name(s) and append to the index
                save_fname = parse_fname(file.name)
                self.files.append(save_fname)
                self.file_paths.append(file)

    def __len__(self) -> int:
        """Returns the number of files in the index."""
        return len(self.files)

    def match_files_by_config(self,
                        config: ClassifierConfig
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

    def __init__(self,
                config: ClassifierConfig,
                load_dir: str | Path = "./src/classifier/out/"
            ) -> None:
        self.config: ClassifierConfig = config
        self.load_dir: Path = Path(load_dir)
        # initialise empty attributes
        self.files: list[Path] = []
        self.book_probas: BookProbas | None = None
        self.book_verses: list[BookVerses] | None = None
        self.result_stats: list[ResultStats] = []

    def load_results(self, save_files: list[SavefileName]) -> None:
        """A utility to load data into the model."""
        for file in save_files:
            if file.is_mislabel:
                # skip the file(s) for inspecting mislabelled instances
                continue

            if file.is_total_proba:
                self.book_probas = load_book_probas(
                                        file.origin, file, self.load_dir)
            elif file.is_clf_summary:
                # load ResultStats from the summary (``classifier_stats``) json
                # file(s)
                pass
            else:
                # if the file is a normal CSV listing verses and their probas
                self.book_verses = load_preds(file.origin, file,
                                        str(self.load_dir.resolve()))
