"""Entry point for the frozen app.

PyInstaller runs its entry script as a loose module, so `penplot/__main__.py`
with its relative imports cannot be it - there is no parent package to be
relative to.  This imports the package properly and calls into it.
"""

import sys

from penplot.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
