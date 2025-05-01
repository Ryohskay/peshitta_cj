Usage Guide
===========

Prerequisites
-------------

1. Install ``poetry``
2. Install dependency packages by ``poetry install`` (at the project root)

Running classifiers
-------------------

To run the classifier code, execute ``poetry run python -m src.classifier.{module-name}`` at the project root directory.
This is the directory named "peshitta_cj" if you cloned this repository.

.. note::
    This may take many hours. If you just want to play around with the code,
    you might want to comment out some ``for`` loops in the files.

Running the UI
--------------

Run the UI by executing: ``poetry run python -m src.ui_web.app``

Building the Sphinx documentation
---------------------------------

To build the Sphinx documentation, ``cd`` to the ``docs/`` directory and run ``make html``.

You can use `make autobuild` to run a web server that automatically rebuilds upon source file changes.
But for this, you need to install `sphinx-autobuild` dependency yourself: ``pip install -U sphinx-autobuild``

In order to build the documentation locally, you also need the ReadTheDoc theme: ``pip install -U sphinx-rtd-theme``