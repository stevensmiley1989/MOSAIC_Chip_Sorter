MOSAIC_Chip_Sorter
========
'''
Created by Steven Smiley 3/20/2022

MOSAIC_Chip_Sorter.py creates mosaics based on the PascalVOC XML annotation files (.xml) generated with labelImg 
or whatever method made with respect to a corresponding JPEG (.jpg) image.  

MOSAIC_Chip_Sorter.py allows easy interfacing with annotation tool, labelImg.py (git clone https://github.com/tzutalin/labelImg.git).

User can quickly find mislabeled objects and alter them on the fly.

        'Step 1a. Open Annotations Folder ',                *required
        'Step 1b. Open JPEGImages Folder',                  *required
        'Step 2. Create DF',                               *first time is a must, optional after.  This creates the pandas DataFrame (.pkl) of your annotations to sort/index with.  THIS MIGHT TAKE A WHILE DEPENDING ON DATA.
        'Step 3. Load DF',                                 *required.  This loads the pandas DataFrame (.pkl) file.
        'Step 4.  Analyze Target',                          *required.  This lets you pick which class to inspect and find bad labels
        'Step 5.  move_fix',                                *required.  This must be done before using labelImg.
        'Step 6.  Fix with labelImg.py',                    *optional.  This lets you fix the annotation associated with the image using labelImg.
        'Step 7.  merge_fix'                                *optional.  This is what transfers fixed annotations over the existing annotation when you are satisified with Step 5.
        'Step 8.  clear_fix',                               *optional.  You can use this after you have done Step 4 at anytime.
        'Step 9.  clear_checked',                           #optional.  As you look through Mosaics, you won't see previous "checked" chips.  If you want to reset to see all, press this button.

'''

MOSAIC_Chip_Sorter is a graphical image annotation tool.

It is written in Python and uses Tkinter for its graphical interface.

As chips are selected, their corresponding Annotation/JPEG files are put in new directories with "...to_fix".  These are used later with LabelImg.py or whatever label tool to update. 


Installation
------------------

Ubuntu Linux/Mac
^^^^^^^^^^^^

Python 3 + Tkinter

.. code:: shell

    sudo pip3 install -r requirements.txt
    python3 MOSAIC_Chip_Sorter.py


Hotkeys
~~~~~~~

+--------------------+--------------------------------------------+
| n                  | Next Batch of Images in Mosaics            |
+--------------------+--------------------------------------------+
| q                  | Close the Mosaic Images                    |
+--------------------+--------------------------------------------+

