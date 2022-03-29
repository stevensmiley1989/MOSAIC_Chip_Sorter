# MOSAIC_Chip_Sorter
## Repository by Steven Smiley 3/20/2022
![MOSAIC_Chip_Sorter.py](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Main.png)
MOSAIC_Chip_Sorter.py is a labeling annotation tool that allows one to identify & fix mislabeled annotation data (chips) in their dataset through the 
use of MOSAICs and UMAP quickly. 

MOSAIC_Chip_Sorter.py creates mosaics based on the PascalVOC XML annotation files (.xml) generated with labelImg 
or whatever method made with respect to a corresponding JPEG (.jpg) image.  

Uniform Manifold Approximation and Projection (UMAP, https://umap-learn.readthedocs.io/en/latest/) is a dimension reduction technique that can be used for visualization of the data into clusters. 

MOSAIC_Chip_Sorter.py allows easy interfacing with annotation tool, labelImg.py (git clone https://github.com/tzutalin/labelImg.git).


It is written in Python and uses Tkinter for its graphical interface.

As chips are selected, their corresponding Annotation/JPEG files are put in new directories with "...to_fix".  These are used later with LabelImg.py or whatever label tool to update. 


Installation
------------------

Ubuntu Linux
~~~~~~~

Python 3 + Tkinter

.. code:: shell

    sudo pip3 install -r requirements.txt
    python3 MOSAIC_Chip_Sorter.py
~~~~~~~

Mac
~~~~~~~

Python 3 + Tkinter

.. code:: shell

    sudo pip3 install -r requirements_Mac.txt
    python3 MOSAIC_Chip_Sorter.py
~~~~~~~

Hotkeys
------------------

MOSAIC Hotkeys
~~~~~~~
+--------------------+--------------------------------------------+
| KEY                | DESCRIPTION                                |
+--------------------+--------------------------------------------+
| n                  | Next Batch of Images in Mosaics            |
+--------------------+--------------------------------------------+
| q                  | Close the Mosaic Images                    |
+--------------------+--------------------------------------------+

~~~~~~~


UMAP Hotkeys
~~~~~~~
+--------------------+--------------------------------------------+
| KEY                | DESCRIPTION                                |
+--------------------+--------------------------------------------+
| n                  | Closes inspected image if open.            |
+--------------------+--------------------------------------------+
| f                  | Adds inspected image to the fix list.      |
+--------------------+--------------------------------------------+
| q                  | Closes the window.                         |
+--------------------+--------------------------------------------+
| Esc                | Closes the window.                         |
+--------------------+--------------------------------------------+
| d                  | Deletes the object from annotation file.   |
+--------------------+--------------------------------------------+
| 1,2,3,etc          | Changes label to key index displayed.      |
+--------------------+--------------------------------------------+

~~~~~~~
## Contact-Info<a class="anchor" id="4"></a>

Feel free to contact me to discuss any issues, questions, or comments.

* Email: [stevensmiley1989@gmail.com](mailto:stevensmiley1989@gmail.com)
* GitHub: [stevensmiley1989](https://github.com/stevensmiley1989)
* LinkedIn: [stevensmiley1989](https://www.linkedin.com/in/stevensmiley1989)
* Kaggle: [stevensmiley](https://www.kaggle.com/stevensmiley)

### License <a class="anchor" id="5"></a>
MIT License

Copyright (c) 2022 Steven Smiley

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*I am providing code and resources in this repository to you under an open source license.  Because this is my personal repository, the license you receive to my code and resources is from me and not my employer. 


User can quickly find mislabeled objects and alter them on the fly.
| STEPS| 
|--------------------------------------------------------------------------------------------------------------|
| ![1.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step1.png)| 
| ![2.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step2.png)| 
| ![3.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step3.png)| 
| ![4a.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step4a.png)|  
| ![4b.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step4b.png)|  
| ![4c.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step4c.png)|  
| ![4d.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step4d.png)|  
| ![4e.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step4e.png)|  
| ![4f.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step4f.png)|  
| ![4g.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step4g.png)|  
| ![5a.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step5a.png)|
| ![5b.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step5b.png)|
| ![5c.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step5c.png)|
| ![5d.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step5d.png)|
| ![5e.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step5e.png)|
| ![6a.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step6a.png)|
| ![6b.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step6b.png)|
| ![6c.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step6c.png)|
| ![6d.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step6d.png)|
| ![7a.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step7a.png)|
| ![7b.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step7b.png)|
| ![8a.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step8a.png)|
| ![8b.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step8b.png)|
| ![9a.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step9a.png)|
| ![9b.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step9b.png)|
| ![9c.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step9c.png)|
| ![10a.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step10a.png)|
| ![10b.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step10b.png)|
| ![11a.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step11a.png)|
| ![11b.](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step11b.png)|


