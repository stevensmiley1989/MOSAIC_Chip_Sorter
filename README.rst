# MOSAIC_Chip_Sorter
## Repository by Steven Smiley 3/20/2022

MOSAIC_Chip_Sorter.py creates mosaics based on the PascalVOC XML annotation files (.xml) generated with labelImg 
or whatever method made with respect to a corresponding JPEG (.jpg) image.  

MOSAIC_Chip_Sorter.py allows easy interfacing with annotation tool, labelImg.py (git clone https://github.com/tzutalin/labelImg.git).

User can quickly find mislabeled objects and alter them on the fly.
# STEPS 
* [1. Step 1a. Open Annotations Folder](https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter/blob/main/misc/Step1.png),                *required
        'Step 1b. Open JPEGImages Folder',                  *required
        'Step 2. Create DF',                               *first time is a must, optional after.  This creates the pandas DataFrame (.pkl) of your annotations to sort/index with.  THIS MIGHT TAKE A WHILE DEPENDING ON DATA.
        'Step 3. Load DF',                                 *required.  This loads the pandas DataFrame (.pkl) file.
        'Step 4.  Analyze Target',                          *required.  This lets you pick which class to inspect and find bad labels
        'Step 5.  move_fix',                                *required.  This must be done before using labelImg.
        'Step 6.  Fix with labelImg.py',                    *optional.  This lets you fix the annotation associated with the image using labelImg.
        'Step 7.  merge_fix'                                *optional.  This is what transfers fixed annotations over the existing annotation when you are satisified with Step 5.
        'Step 8.  clear_fix',                               *optional.  You can use this after you have done Step 4 at anytime.
        'Step 9.  clear_checked',                           #optional.  As you look through Mosaics, you won't see previous "checked" chips.  If you want to reset to see all, press this button.


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
~~~~~~~
+--------------------+--------------------------------------------+
| KEY                | DESCRIPTION                                |
+--------------------+--------------------------------------------+
| n                  | Next Batch of Images in Mosaics            |
+--------------------+--------------------------------------------+
| q                  | Close the Mosaic Images                    |
+--------------------+--------------------------------------------+

~~~~~~~
## 4 Contact-Info<a class="anchor" id="4"></a>

Feel free to contact me to discuss any issues, questions, or comments.

* Email: [stevensmiley1989@gmail.com](mailto:stevensmiley1989@gmail.com)
* GitHub: [stevensmiley1989](https://github.com/stevensmiley1989)
* LinkedIn: [stevensmiley1989](https://www.linkedin.com/in/stevensmiley1989)
* Kaggle: [stevensmiley](https://www.kaggle.com/stevensmiley)

### 5 License <a class="anchor" id="5"></a>
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
