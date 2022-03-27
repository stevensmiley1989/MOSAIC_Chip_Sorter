import setuptools
from sys import platform as _platform
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
# OS specific settings
SET_REQUIRES = []
if _platform == "linux" or _platform == "linux2":
   # linux
   print('linux')
elif _platform == "darwin":
   # MAC OS X
   SET_REQUIRES.append('tkmacosx')
setuptools.setup(
name = "MOSAIC_Chip_Sorter",
version = "0.0.1",
author = "Steven Smiley",
author_email = "stevensmiley1989@gmail.com",
description = "MOSAIC_Chip_Sorter.py is a labeling tool.  It creates clickable/fixable mosaics based on the PascalVOC XML annotation files (.xml).  It also uses UMAP to cluster chips for fixing.",   
long_description = long_description,
long_description_content_type = "text/markdown",
url = "https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter",
project_urls ={
    "Bug Tracker":"https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/issues"},
classifiers =[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",],
package_dir ={"":"."},
packages =setuptools.find_packages(where='.'),
setup_requires=SET_REQUIRES,
python_requires = ">=3.7",
)