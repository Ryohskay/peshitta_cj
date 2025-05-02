# Peshitta CJ

## Purpose

Categorise verses from OT Peshitta (a *Classical Syriac* Translation of the Old Testament) with unknown authorship, into books translated by Jews and by Christians, through training ML classifiers to assign probabilities.

## See also

- User Manual
- Maintenance Manual

## Repository Structure

- `docs/`: Sphinx directory
- `src/`: python scripts and utility files 
  - `shared/`: utilities and constants to be shared among multiple modules
  - `scraper/`: code to extract Peshitta texts from the [CAL (Comprehensive Aramaic Lexicon, Stephen A. Kaufman et al.)](https://cal.huc.edu/)
  - `classifier/`: code to build and evaluate classifiers for categorising Peshitta verses.
- `notebooks/`: Jupyter notebook files for experiments

## System Design

- The code and system will be based on the Latin Alphabet transliteration (*romanisation*) used on the CAL.
- This implementation ignores vocalisation in NT Peshitta, since it's unnecessary for the purpose of this project.
  - Vocalisation is not indicated on the CAL OT Peshitta.
  - Vocalisation of OT Peshitta is likely to include works done much later in the history, which may negatively impact the objective reliability of the categorisation by this system.

## Author

This project is a work by Ryosuke Nagata (to be) submitted as his dissertation project for Computing, MA., Hons. at the University of Aberdeen.
