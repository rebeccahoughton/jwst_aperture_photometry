import sys
from importlib.metadata import version, PackageNotFoundError

# Check package version
try:
    __version__ = version("phot")
except PackageNotFoundError:
    __version__ = "unknown"

if sys.version_info < (3, 11):
    raise ImportError("photometry code requires Python 3.11 and above.")

# python's module setup seems unnecessarily complicated. 
# import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# All function imports
from rphot.io import get_file

__all__ = [
    "get_file",
]
