# Peshitta CJ

## Purpose

Categorise the OT Peshitta (a *Classical Syriac* Translation of the Old Testament) into:

1. books translated from Hebrew by Jews and
2. books translated from Hebrew by Christians.

## Repository Structure

- `/scraper/`: code to extract Peshitta texts from the [CAL (Comprehensive Aramaic Lexicon, Stephen A. Kaufman et al.)](https://cal.huc.edu/)

## System Design

- The code and system will be based on the Latin Alphabet transliteration (*romanisation*) used on the CAL.
- This implementation ignores vocalisation in NT Peshitta, since it's unnecessary for the purpose of this project.
  - Vocalisation is not indicated on the CAL OT Peshitta.
  - Vocalisation of OT Peshitta is likely to include works done much later in the history, which may negatively impact the objective reliability of the categorisation by this system.

## Author

This project is a work by Ryosuke Nagata (to be) submitted as his dissertation for Computing, MA., Hons. at the University of Aberdeen.
