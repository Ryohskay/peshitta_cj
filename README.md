# Peshitta CJ

## Purpose

Categorise the OT Peshitta (a *Classical Syriac* Translation of the Old Testament) into:

1. books translated from Hebrew by Jews and
2. books translated from Greek (with some help of Hebrew references) by Christians.

## Repository Structure

- `./docs/`: Sphinx directory
- `./src/`: python scripts and utility files 
 - `scraper/`: code to extract Peshitta texts from the [CAL (Comprehensive Aramaic Lexicon, Stephen A. Kaufman et al.)](https://cal.huc.edu/)
 - `classifier/`: code to build and evaluate classifiers for categorising Peshitta verses.1
- `./cite_tf/`: Jupyter notebook files for experiments

## System Design

- The code and system will be based on the Latin Alphabet transliteration (*romanisation*) used on the CAL.
- This implementation ignores vocalisation in NT Peshitta, since it's unnecessary for the purpose of this project.
  - Vocalisation is not indicated on the CAL OT Peshitta.
  - Vocalisation of OT Peshitta is likely to include works done much later in the history, which may negatively impact the objective reliability of the categorisation by this system.

## Author

This project is a work by Ryosuke Nagata (to be) submitted as his dissertation project for Computing, MA., Hons. at the University of Aberdeen.

## Notes
- To run the classifier code, execute `python3 -m classifier.{module-name}` at the `./src/` directory.
- To build the Sphinx documentation, run `make html` at the `./docs/` directory.
 - You can use `make autobuild` to run a web server that automatically rebuilds upon source file changes.
 - But for this, you need to install `sphinx-autobuild` dependency yourself: `pip install -U sphinx-autobuild`
 - In order to build the documentation locally, you also need the ReadTheDoc theme: `pip install -U sphinx-rtd-theme` 
