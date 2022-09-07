'''
MOSAIC_Chip_Sorter
========
Created by Steven Smiley 3/20/2022

MOSAIC_Chip_Sorter.py is a labeling annotation tool that allows one to identify & fix mislabeled annotation data (chips) in their dataset through the 
use of MOSAICs and UMAP quickly. 

MOSAIC_Chip_Sorter.py creates mosaics based on the PascalVOC XML annotation files (.xml) generated with labelImg 
or whatever method made with respect to a corresponding JPEG (.jpg) image.  

Uniform Manifold Approximation and Projection (UMAP, https://umap-learn.readthedocs.io/en/latest/) is a dimension reduction technique that can be used for visualization of the data into clusters.

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
        'Step 10.  Breakup df',                             #optional.  If you want to use UMAP on over 1000 or so annotations at a time, then this will break it up in chunks of 1000 for easy sorting and merging later.
        'Step 11.  UMAP',                                   #optional.  Lets you use UMAP to pick annotations out of the files to view in real-time.  You can add items you find to "fix" category.  Change labels or delete objects on the fly.
        'Step 12.  Open Originals with LabelIMG;
        'Step 13.  Plot DX vs DY;
        'Step 14.  Filter by DX & DY;
        'Step 15.  Make Chips;    
        'Step 16.  Change Labels          
MOSAIC_Chip_Sorter is a graphical image annotation tool.

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
| y                  | Select all objects in Mosaic               |
+--------------------+--------------------------------------------+
| u                  | Unselect all objects in Mosaic             |
+--------------------+--------------------------------------------+
| t                  | Drop Down Menu for changing name of object |
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
+--------------------+--------------------------------------------+
| m                  | Creates MOSAIC to fix labels of selected.  |
+--------------------+--------------------------------------------+

~~~~~~~

'''
import argparse
from chunk import Chunk
import multiprocessing
import sys
from sys import platform as _platform
import pandas as pd
from tqdm import tqdm
import os
import matplotlib

matplotlib.rcParams['interactive'] == True

import matplotlib.pyplot as plt

from functools import partial
from threading import Thread
import shutil
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import numpy as np
import functools
import time
import PIL
from PIL import Image, ImageTk
Image.MAX_IMAGE_PIXELS = None #edit sjs 5/28/2022
from PIL import ImageDraw
from PIL import ImageFont
import tkinter as tk
from tkinter import Toplevel, ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter.tix import Balloon
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import umap 
from pprint import pprint
import random
import cv2
try:
    from libs import SAVED_SETTINGS as DEFAULT_SETTINGS
except:
    from libs import DEFAULT_SETTINGS 

if _platform=='darwin':
    import tkmacosx
    from tkmacosx import Button as Button
    open_cmd='open'
    matplotlib.use('MacOSX')
else:
    from tkinter import Button as Button
    if _platform.lower().find('linux')!=-1:
        open_cmd='xdg-open'
    else:
        open_cmd='start'
root_tk=tk.Tk()
ROOT_H=int(root_tk.winfo_screenheight()*0.95)
ROOT_W=int(root_tk.winfo_screenwidth()*0.95)
print('ROOT_H',ROOT_H)
print('ROOT_W',ROOT_W)
from pandastable import Table, TableModel
class TestApp(tk.Frame):
    def __init__(self, parent, filepath):
        super().__init__(parent)
        self.table = Table(self, showtoolbar=True, showstatusbar=True)
        if filepath.find('.csv') and os.path.exists(filepath)==False:
            pklpath=filepath.replace('.csv','.pkl')
            df_i=pd.read_pickle(pklpath)
            df_i.to_csv(filepath,index=None)
        self.table.importCSV(filepath)
        #self.table.load(filepath)
        #self.table.resetIndex()
        self.table.show()
def remove_directory(dir):
    #dir = 'path/to/dir'
    for files in os.listdir(dir):
        path = os.path.join(dir, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)
    os.rmdir(dir)
def create_img_list(path_change=None):
    '''path_change should be in the JPEGImages directory'''
    if path_change:
        return_dir=os.getcwd()
        os.chdir(path_change)
    else:
        return_dir=os.getcwd()
    f=open('img_list.txt','w')
    imgs=os.listdir()
    imgs=[w for w in imgs if w.find('.jpg')!=-1]
    cwd=os.getcwd()
    img_list=[os.path.join(cwd,w) for w in imgs]
    tmp=[f.writelines(w+'\n') for w in img_list]
    f.close()
    os.chdir(return_dir)
def create_imgs_from_video(Annotations_path,JPEGImages_path,path_movie=None,fps='1/2'):
    '''path_change should be the MOV or mp4 path '''
    time_i=str(time.time())
    time_i=time_i.split('.')[0]
    if str(type(fps)).find('str')==-1:
        print('This is not a string for the fps! {}'.format(str(type(fps))))
    elif path_movie.lower().find('.mp4')!=-1 or path_movie.lower().find('.mov')!=-1:
        return_dir=os.getcwd()
        
        basepath=os.path.dirname(path_movie)
        files=os.listdir(return_dir)
        if 'ffmpeg.exe' in files:
            shutil.copy('ffmpeg.exe',os.path.dirname(basepath))
            print('Copied ffmpeg.exe to {}'.format(basepath))
        else:
            print('Depending on having ffmpeg installed')
        os.chdir(basepath)
        print(basepath)

        movie_i_name=os.path.basename(path_movie).split('.')[0]
        folders_in_basepath=os.listdir(basepath)
        folders_in_basepath=[w for w in folders_in_basepath if os.path.isdir(os.path.join(basepath,w))]
        JPEGImages_path=os.path.join(basepath,'JPEGImages')
        Annotations_path=os.path.join(basepath,'Annotations')
        if 'Annotations' not in folders_in_basepath:
            os.makedirs(Annotations_path)
        else:
            #os.system('mv {} {}'.format(Annotations_path,Annotations_path+'_backup_{}'.format(time_i)))
            Annotations_path=Annotations_path+'_'+time_i
            os.makedirs(Annotations_path)
        if 'JPEGImages' not in folders_in_basepath:
            os.makedirs(JPEGImages_path)  
        else:
            #os.system('mv {} {}'.format(JPEGImages_path,JPEGImages_path+'_backup_{}'.format(time_i)))
            JPEGImages_path=JPEGImages_path+'_'+time_i
            os.makedirs(JPEGImages_path)


        os.system('ffmpeg -i {} -qscale:v 2 -vf fps={} {}_fps{}_%08d.jpg'.format(path_movie,fps,os.path.join(JPEGImages_path,movie_i_name),fps.replace('/','d').replace('.','p')))
        os.chdir(return_dir)
        print('creating img list')
        create_img_list(JPEGImages_path)
        print('finished creating img list')
        return Annotations_path,JPEGImages_path
    else:
        print('This is not a valid movie file.  Needs to be .mp4 or .MOV.  \n Provided: {}'.format(path_movie))
        return Annotations_path,JPEGImages_path
def FIND_REPLACE_TARGETS(targets_dic,path_annotations):
	targets_keep=targets_dic.values()
	print('targets_keep',targets_keep)
	#targets_keep=['transporter9']
	#path_annotations="/media/steven/Elements/Drone_Videos/Combined_Transporter9_withVisDrone_noanno/Annotations"
	path_annotations_new=path_annotations

	annotation_files=os.listdir(path_annotations)
	annotation_files=[w for w in annotation_files if w.find('.xml')!=-1]
	start=False
	keep=True
	for file in tqdm(annotation_files):
		file_og=os.path.join(path_annotations,file)
		file_new=os.path.join(path_annotations_new,file)
		f=open(file_og,'r')
		f_read=f.readlines()
		if len(f_read)==1:
			f_read=f_read[0].replace('><','>\n<')
		f.close()
		f_new=[]
		for i,line in enumerate(f_read):
			if line.find('<object>')!=-1:
				start=True
				name=f_read[i+1].split('<name>')[1].split('</name>')[0].replace('\n','').strip()
				name_og=name.strip()
				if name in list(targets_dic.keys()):
					name=targets_dic[name]
				if name in list(targets_keep):
					#print('BEFORE: \t',f_read[i+1])
					keep=True
					f_read[i+1]=f_read[i+1].replace(name_og.strip(),name.strip()) #find/replace label
					#print('AFTER: \t',f_read[i+1])
				else:
					keep=False
			if line.find('</object>')!=-1:
				start=False
				if keep:
					f_new.append(line)
				else:
					pass
			elif start and keep:
				f_new.append(line)
			elif start and keep==False:
				pass
			else:
				f_new.append(line)
		f=open(file_new,'w')
		tmp=[f.writelines(w) for w in f_new]
		f.close()
class popupWindow(object):
    def __init__(self,master):
        self.top=tk.Toplevel(master)
        self.top.geometry( "{}x{}".format(ROOT_W//2,ROOT_H//2) )
        self.top.configure(background = 'black')
        self.l=tk.Label(self.top,text="Enter new label for item",bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.l.pack()
        self.e=tk.Entry(self.top)
        self.e.pack()
        self.b=Button(self.top,text='Ok',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.pack()
    def cleanup(self):
        self.value=self.e.get()
        self.top.destroy()

class popupWindow_REMOVEBLANKS(object):
    def __init__(self,master,blank_annos,blank_jpegs):
        self.top=tk.Toplevel(master)
        self.top.geometry( "{}x{}".format(ROOT_W//2,ROOT_H//2) )
        self.top.configure(background = 'black')
        self.l=Button(self.top,text="Yes",command=self.cleanup_yes,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.l.pack()
        self.e=Button(self.top,text="No",command=self.cleanup_no,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.e.pack()
        self.clicked_anno=tk.StringVar()
        if len(blank_annos)>0:
            self.clicked_anno.set(blank_annos[0])
        else:
            blank_annos=['No Blank Annotations Found']
            self.clicked_anno.set(blank_annos[0])
        self.opt_anno=tk.OptionMenu(self.top,self.clicked_anno,*blank_annos)
        self.opt_anno.pack()
        self.clicked_jpeg=tk.StringVar()
        if len(blank_jpegs)>0:
            self.clicked_jpeg.set(blank_jpegs[0])
        else:
            blank_jpegs=['No Blank JPEGs Found']
            self.clicked_jpeg.set(blank_jpegs[0])
        self.opt_jpegs=tk.OptionMenu(self.top,self.clicked_jpeg,*blank_jpegs)
        self.opt_jpegs.pack()
    def cleanup_yes(self):
        self.value="Yes"
        self.top.destroy()
    def cleanup_no(self):
        self.value="No"
        self.top.destroy()

class popupWindow_CHIPS(object):
    def __init__(self,master,unique_labels):
        self.top=tk.Toplevel(master)
        self.top.geometry( "{}x{}".format(ROOT_W//2,ROOT_H//2) )
        self.top.configure(background = 'black')
        self.l=Button(self.top,text="Submit",command=self.cleanup_yes,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.l.grid(row=1,column=1,sticky='se')
        self.e=Button(self.top,text="Cancel",command=self.cleanup_no,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.e.grid(row=1,column=2,sticky='sw')
        self.style4=ttk.Style()
        self.style4.configure('Normal.TCheckbutton',
                             background='green',
                             foreground='black')
        self.checkm_vars={}
        self.checkm_buttons={}
        for i,label in enumerate(unique_labels):
            self.checkm_vars[label]=tk.IntVar()
            self.checkm_vars[label].set(1)
            self.checkm_buttons[label]=ttk.Checkbutton(self.top, style='Normal.TCheckbutton',text=label,variable=self.checkm_vars[label],onvalue=1, offvalue=0)
            self.checkm_buttons[label].grid(row=i+1,column=3,sticky='sw')     

    def cleanup_yes(self):
        CHIP_LIST=[]
        for label_i,value_i in self.checkm_vars.items():
            if value_i.get()==1:
                CHIP_LIST.append(label_i)
        self.value=CHIP_LIST
        self.top.destroy()
    def cleanup_no(self):
        CHIP_LIST=[]
        self.value=CHIP_LIST
        self.top.destroy()

class popupWindow_CHIPS_CLEAR(object):
    def __init__(self,master,path_chips):
        self.top=tk.Toplevel(master)
        self.top.geometry( "{}x{}".format(ROOT_W//2,ROOT_H//2) )
        self.top.configure(background = 'black')
        self.l=Button(self.top,text="CLEAR EXISTING CHIPS?",command=self.cleanup_yes,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.l.grid(row=1,column=1,sticky='se')
        self.e=Button(self.top,text="KEEP EXISTING CHIPS?",command=self.cleanup_no,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.e.grid(row=1,column=2,sticky='sw')  
        os.system(f'xdg-open {path_chips}')  
    def cleanup_yes(self):
        self.value=True
        self.top.destroy()
    def cleanup_no(self):
        self.value=False
        self.top.destroy()

class popupWindow_KEEPCLASS(object):
    def __init__(self,master,new_df):
        self.top=tk.Toplevel(master)
        self.top.geometry( "{}x{}".format(ROOT_W//2,ROOT_H//2) )
        self.top.configure(background = 'black')
        self.l=Button(self.top,text="Submit",command=self.cleanup_yes,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.l.grid(row=1,column=1,sticky='se')
        self.e=Button(self.top,text="Cancel",command=self.cleanup_no,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.e.grid(row=1,column=2,sticky='sw')
        self.style4=ttk.Style()
        self.style4.configure('Normal.TCheckbutton',
                             background='green',
                             foreground='black')
        self.checkm_vars={}
        self.checkm_buttons={}
        self.new_df=new_df
        unique_labels=self.new_df['label_i'].unique()

        self.label_class=tk.Label(self.top,text='Classes to keep',bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg,font=('Arial 10 underline'))
        self.label_class.grid(row=0,column=3,sticky='se')
        for i,label in enumerate(unique_labels):
            self.checkm_vars[label]=tk.IntVar()
            self.checkm_vars[label].set(1)
            self.checkm_buttons[label]=ttk.Checkbutton(self.top, style='Normal.TCheckbutton',text=label,variable=self.checkm_vars[label],onvalue=1, offvalue=0)
            self.checkm_buttons[label].grid(row=i+1,column=3,sticky='sw')    

        self.difficulty_label=tk.Label(self.top,text='Difficulty to keep',bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg,font=('Arial 10 underline'))
        self.difficulty_label.grid(row=0,column=5,sticky='se')
        self.diff_vars={}
        self.diff_buttons={}
        self.diff_vars['0']=tk.IntVar()
        self.diff_vars['0'].set(1)
        self.diff_vars['1']=tk.IntVar()
        self.diff_vars['1'].set(1)
        self.diff_buttons['0']=ttk.Checkbutton(self.top, style='Normal.TCheckbutton',text='0 = not difficult',variable=self.diff_vars['0'],onvalue=1, offvalue=0)
        self.diff_buttons['0'].grid(row=1,column=5,sticky='se')
        self.diff_buttons['1']=ttk.Checkbutton(self.top, style='Normal.TCheckbutton',text='1 = difficult',variable=self.diff_vars['1'],onvalue=1, offvalue=0)
        self.diff_buttons['1'].grid(row=2,column=5,sticky='se')



            
    def cleanup_yes(self):
        KEEPCLASS_LIST=[]
        for label_i,value_i in self.checkm_vars.items():
            if value_i.get()==1:
                KEEPCLASS_LIST.append(label_i)
        self.unique_xmls=list(self.new_df[self.new_df['label_i'].isin(KEEPCLASS_LIST)]['path_anno_i'].unique())
        self.new_df=self.new_df[self.new_df['path_anno_i'].isin(self.unique_xmls)].reset_index().drop('index',axis=1)
        keep_diff_list=[]
        if self.diff_vars['0'].get()==1:
            keep_diff_list.append('0')
        if self.diff_vars['1'].get()==1:
            keep_diff_list.append('1')
        self.new_df=self.new_df[self.new_df['difficulty'].isin(keep_diff_list)].reset_index().drop('index',axis=1)
        self.value=self.new_df
        self.top.destroy()
    def cleanup_no(self):
        KEEPCLASS_LIST=[]
        self.value=pd.DataFrame()
        self.top.destroy()

class popupWindow_BLACKPATCHES(object):
    def __init__(self,master,df,df_filename,df_filename_csv):
        self.top=tk.Toplevel(master)
        self.top.geometry( "{}x{}".format(ROOT_W//2,ROOT_H//2) )
        self.top.configure(background = 'black')
        self.l=Button(self.top,text="Submit",command=self.cleanup_yes,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.l.grid(row=1,column=1,sticky='se')
        self.e=Button(self.top,text="Cancel",command=self.cleanup_no,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.e.grid(row=1,column=2,sticky='sw')
        self.style4=ttk.Style()
        self.style4.configure('Normal.TCheckbutton',
                             background='green',
                             foreground='black')
        self.checkm_vars={}
        self.checkm_buttons={}
        self.new_df=df.copy()
        self.df=df.copy()
        self.df_filename=df_filename
        self.df_filename_csv=df_filename_csv
        unique_labels=self.new_df['label_i'].unique()

        for i,label in enumerate(unique_labels):
            self.checkm_vars[label]=tk.IntVar()
            self.checkm_vars[label].set(1)
            self.checkm_buttons[label]=ttk.Checkbutton(self.top, style='Normal.TCheckbutton',text=label,variable=self.checkm_vars[label],onvalue=1, offvalue=0)
            self.checkm_buttons[label].grid(row=i+1,column=3,sticky='sw')     
            
    def cleanup_yes(self):
        BLACKPATCHES_LIST=[]
        for label_i,value_i in self.checkm_vars.items():
            if value_i.get()==1:
                BLACKPATCHES_LIST.append(label_i)
        self.new_df=self.new_df[self.new_df['label_i'].isin(BLACKPATCHES_LIST)]
        self.bad_index_list=[]
        for row in tqdm(range(len(self.new_df))):
            self.index_bad=self.new_df.iloc[row].name
            path_jpeg_i=self.new_df['path_jpeg_i'].iloc[row]
            xmin=self.new_df['xmin'].iloc[row]
            xmax=self.new_df['xmax'].iloc[row]
            ymin=self.new_df['ymin'].iloc[row]
            ymax=self.new_df['ymax'].iloc[row]
            self.img_i=cv2.imread(path_jpeg_i)
            if len(self.img_i.shape)==3:
                self.img_i[ymin:ymax,xmin:xmax,:]=0.
            else:
                self.img_i[ymin:ymax,xmin:xmax]=0.
            cv2.imwrite(path_jpeg_i,self.img_i)

            try:
            
                xmin=str(xmin)
                ymin=str(ymin)
                xmax=str(xmax)
                ymax=str(ymax)

                label_k=self.df['label_i'][self.index_bad]
                path_anno_bad=self.df['path_anno_i'][self.index_bad]

                assert path_jpeg_i==self.df['path_jpeg_i'][self.index_bad]
                f=open(path_anno_bad,'r')
                f_read=f.readlines()
                f.close()
                f_new=[]
                start_line=0
                end_line=0

                self.bad_index_list.append(self.index_bad)
                for ii,line in enumerate(f_read):
                    if line.find(label_k)!=-1:   
                        combo_i=f_read[ii-1:]
                        combo_i="".join([w.replace('\n','') for w in combo_i])
                        combo_i=combo_i.split('<object>')

                        #print(len(combo_i),combo_i)
                        if len(combo_i)>1:
                            if combo_i[1].find(xmin)!=-1 and combo_i[1].find(xmax)!=-1 and combo_i[1].find(ymin)!=-1 and combo_i[1].find(ymax)!=-1:
                                start_line=ii-1
                                #for jj,line_j in enumerate(f_read[ii:]):
                                for jj,line_j in enumerate(combo_i[1].split('</')):
                                    if line_j.find('object>')!=-1:
                                        end_line=jj+ii
                                        print('found label_k',label_k)
                                        print('deleting bounding box')
                                        print('xmin',xmin)
                                        print('xmax',xmax)
                                        print('ymin',ymin)
                                        print('ymax',ymax)
                                f_new.append(line)
                            else:
                                f_new.append(line)
                        else:
                            f_new.append(line)
                    else:
                        f_new.append(line)
                if end_line!=0:

                    f_new=f_new[:start_line]+f_new[end_line+1:]
                    try:
                        f_new[0].find('annotation')!=-1
                    except:
                        pprint(f_read)
                        assert f_new[0].find('annotation')!=-1
                #f_new=f_new[:start_line]+f_new[end_line+1:]
                f=open(path_anno_bad,'w')
                [f.writelines(w) for w in f_new]
                f.close()  

            except:
                print('This ship has sailed, item not found.')
        print('cleaning  up the dataframe')
        for bad_index in tqdm(self.bad_index_list):
            self.df=self.df.drop(bad_index,axis=0)
        self.df=self.df.reset_index().drop('index',axis=1)
        self.df.to_pickle(self.df_filename)
        self.df.to_csv(self.df_filename_csv,index=None)
        self.value=self.df
        self.top.destroy()
    def cleanup_no(self):
        self.value=self.df
        self.top.destroy()
class popupWindow_FAKEPATCHES(object):
    def __init__(self,master,df,df_filename,df_filename_csv):
        self.top=tk.Toplevel(master)
        self.top.geometry( "{}x{}".format(ROOT_W//2,ROOT_H//2) )
        self.top.configure(background = 'black')
        self.l=Button(self.top,text="Submit",command=self.cleanup_yes,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.l.grid(row=1,column=1,sticky='se')
        self.e=Button(self.top,text="Cancel",command=self.cleanup_no,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.e.grid(row=1,column=2,sticky='sw')
        self.style4=ttk.Style()
        self.style4.configure('Normal.TCheckbutton',
                             background='green',
                             foreground='black')
        self.checkm_vars={}
        self.checkm_buttons={}
        self.new_df=df.copy()
        self.df=df.copy()
        self.df_filename=df_filename
        self.df_filename_csv=df_filename_csv
        unique_labels=self.new_df['label_i'].unique()

        for i,label in enumerate(unique_labels):
            self.checkm_vars[label]=tk.IntVar()
            self.checkm_vars[label].set(1)
            self.checkm_buttons[label]=ttk.Checkbutton(self.top, style='Normal.TCheckbutton',text=label,variable=self.checkm_vars[label],onvalue=1, offvalue=0)
            self.checkm_buttons[label].grid(row=i+1,column=3,sticky='sw')     


    def cleanup_yes(self):
        FAKEPATCHES_LIST=[]
        path_background='FAKE_BACKGROUNDS'
        if os.path.exists(path_background):
            possible_fake_backgrounds=os.listdir(path_background)
            if len(possible_fake_backgrounds)>0:
                possible_fake_backgrounds=[os.path.join(path_background,w) for w in possible_fake_backgrounds if os.path.isfile(os.path.join(path_background,w)) and w.find('.jpg')!=-1]
                if len(possible_fake_backgrounds)>0:
                    for label_i,value_i in self.checkm_vars.items():
                        if value_i.get()==1:
                            FAKEPATCHES_LIST.append(label_i)
                    self.new_df=self.new_df[self.new_df['label_i'].isin(FAKEPATCHES_LIST)]
                    
                    for row in tqdm(range(len(self.new_df))):
                        path_background_i=random.choice(possible_fake_backgrounds)
                        print('path_background_i',path_background_i)
                        path_jpeg_i=self.new_df['path_jpeg_i'].iloc[row]
                        xmin=self.new_df['xmin'].iloc[row]
                        xmax=self.new_df['xmax'].iloc[row]
                        ymin=self.new_df['ymin'].iloc[row]
                        ymax=self.new_df['ymax'].iloc[row]
                        self.img_i=cv2.imread(path_jpeg_i)
                        self.img_i_H=self.img_i.shape[1]
                        self.img_i_W=self.img_i.shape[0]
                        self.img_fake=cv2.imread(path_background_i)
                        self.img_fake=cv2.resize(self.img_fake,(self.img_i_H,self.img_i_W))
                        if len(self.img_i.shape)==3:
                            self.img_fake[ymin:ymax,xmin:xmax,:]=self.img_i[ymin:ymax,xmin:xmax,:]
                        else:
                            self.img_fake[ymin:ymax,xmin:xmax]=self.img_i[ymin:ymax,xmin:xmax]
                        cv2.imwrite(path_jpeg_i,self.img_fake)
                    self.bad_index_list=[]
                    for row in tqdm(range(len(self.df))):
                        self.index_bad=self.df.iloc[row].name
                        path_jpeg_i=self.df['path_jpeg_i'].iloc[row]
                        list_of_acceptable_jpegs=list(self.new_df['path_jpeg_i'].unique())
                        label_k=self.df['label_i'][self.index_bad]
                        if path_jpeg_i not in list_of_acceptable_jpegs:
                            if os.path.exists(path_jpeg_i):
                                os.remove(path_jpeg_i)
                            self.bad_index_list.append(self.index_bad)
                            path_anno_bad=self.df['path_anno_i'][self.index_bad]
                            if os.path.exists(path_anno_bad):
                                os.remove(path_anno_bad)
                        elif label_k in FAKEPATCHES_LIST:
                            pass
                        else:
                            xmin=self.df['xmin'].iloc[row]
                            xmax=self.df['xmax'].iloc[row]
                            ymin=self.df['ymin'].iloc[row]
                            ymax=self.df['ymax'].iloc[row]

                            #try:
                            
                            xmin=str(xmin)
                            ymin=str(ymin)
                            xmax=str(xmax)
                            ymax=str(ymax)

                            
                            path_anno_bad=self.df['path_anno_i'][self.index_bad]

                            assert path_jpeg_i==self.df['path_jpeg_i'][self.index_bad]
                            if os.path.exists(path_anno_bad):
                                f=open(path_anno_bad,'r')
                                f_read=f.readlines()
                                f.close()
                                try:
                                    f_read[0].find('annotation')!=-1
                                except:
                                    pprint(f_read)
                                    assert f_read[0].find('annotation')!=-1
                                f_new=[]
                                start_line=0
                                end_line=0

                                self.bad_index_list.append(self.index_bad)
                                for ii,line in enumerate(f_read):
                                    if line.find(label_k)!=-1:   
                                        combo_i=f_read[ii-1:]
                                        combo_i="".join([w.replace('\n','') for w in combo_i])
                                        combo_i=combo_i.split('<object>')
                                        #print(len(combo_i),combo_i)
                                        #if len(combo_i)>1:
                                        if combo_i[1].find(xmin)!=-1 and combo_i[1].find(xmax)!=-1 and combo_i[1].find(ymin)!=-1 and combo_i[1].find(ymax)!=-1:
                                            start_line=ii-1
                                            for jj,line_j in enumerate(combo_i[1].split('</')):
                                                if line_j.find('object>')!=-1:
                                                    end_line=jj+ii
                                                    print('found label_k',label_k)
                                                    print('deleting bounding box')
                                                    print('xmin',xmin)
                                                    print('xmax',xmax)
                                                    print('ymin',ymin)
                                                    print('ymax',ymax)
                                            f_new.append(line)
                                        else:
                                            f_new.append(line)
                                        #else:
                                        #    f_new.append(line)
                                    else:
                                        f_new.append(line)
                                if end_line!=0:
                                    f_new=f_new[:start_line]+f_new[end_line+1:]

                                try:
                                    f_new[0].find('annotation')!=-1
                                except:
                                    pprint(f_read)
                                    assert f_new[0].find('annotation')!=-1
                                f=open(path_anno_bad,'w')
                                [f.writelines(w) for w in f_new]
                                f.close()

                            #except:
                            #    print('This ship has sailed, item not found.')
                    print('cleaning  up the dataframe')
                    for bad_index in tqdm(self.bad_index_list):
                        self.df=self.df.drop(bad_index,axis=0)
                    self.df=self.df.reset_index().drop('index',axis=1)
                    self.df.to_pickle(self.df_filename)
                    self.df.to_csv(self.df_filename_csv,index=None)
        self.value=self.df
        self.top.destroy()
    def cleanup_no(self):
        self.value=self.df
        self.top.destroy()

class popupWindowDropDown(object):
    def __init__(self,master,dic_i):
        options=[]
        for k,v in dic_i.items():
            row_i="{}: {}".format(v,k)
            options.append(row_i)
        self.top=tk.Toplevel(master)
        self.top.geometry( "{}x{}".format(ROOT_W//2,ROOT_H//2) )
        self.top.configure(background = 'black')
        if _platform=='darwin':
            self.top.lift()
        self.clicked=tk.StringVar()
        self.clicked.set(options[0])
        self.l=tk.OptionMenu(self.top,self.clicked,*options)
        self.l.pack()
        self.b=Button(self.top,text='Submit',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.pack()
        self.e=Button(self.top,text='cancel',command=self.cancel,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.e.pack()
    def cleanup(self):
        self.value=str(self.clicked.get().split(':')[0])
        self.top.destroy()

    def cancel(self):
        #self.value=str(self.clicked.get().split(':')[0])
        self.value='None Selected'
        self.top.destroy()

class popupWindowDropDown_FPS(object):
    def __init__(self,master):
        options=['30','20','10','5','1','1/2','1/5','1/10']
        self.top=tk.Toplevel(master)
        self.top.geometry( "{}x{}".format(ROOT_W//2,ROOT_H//2) )
        self.top.configure(background = 'black')
        if _platform=='darwin':
            self.top.lift()
        self.clicked=tk.StringVar()
        self.clicked.set('1/2')
        self.l=tk.OptionMenu(self.top,self.clicked,*options)
        self.l.pack()
        self.b=Button(self.top,text='Submit',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.pack()
        self.e=Button(self.top,text='cancel',command=self.cancel,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.e.pack()
    def cleanup(self):
        self.value=str(self.clicked.get())
        self.top.destroy()
    def cancel(self):
        #self.value=str(self.clicked.get().split(':')[0])
        self.value='None Selected'
        self.top.destroy()
class popupWindowChangeLabels(object):
    def __init__(self,master,dic_i,path_annotations,df):
        self.path_annotations=path_annotations
        self.df=df
        options=[]
        for k,v in dic_i.items():
            row_i="{}: {}".format(v,k)
            #print(row_i)
            options.append(row_i)
        self.top=tk.Toplevel(master)
        self.top.geometry( "{}x{}".format(ROOT_W,ROOT_H) )
        self.top.configure(background = 'black')
        self.get_update_background_img()
        #if _platform=='darwin':
        self.top.lift()
        self.new_dic={}
        self.new_labels={}
        self.new_Entries={}
        self.new_labels_og=tk.Label(self.top,text="ORIGINAL VALUE",bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg,font=("Arial", 8))
        self.new_labels_og.grid(row=0,column=1,sticky='se')
        self.new_labels_new=tk.Label(self.top,text="NEW VALUE",bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg,font=("Arial", 8))
        self.new_labels_new.grid(row=0,column=2,sticky='s')
        ii=0
        j=0
        for i,(k,v) in enumerate(dic_i.items()):
            i+=1
            if i%25==0:
                ii=0
                j+=2
            ii+=1
            self.new_dic[k]=tk.StringVar()
            self.new_dic[k].set(k)
            self.new_labels[k]=tk.Label(self.top,text=k+":",bg=DEFAULT_SETTINGS.root_bg, fg=DEFAULT_SETTINGS.root_fg)
            self.new_labels[k].grid(row=ii+1,column=1+j,sticky='ne')
            self.new_Entries[k]=tk.Entry(self.top,textvariable=self.new_dic[k])
            self.new_Entries[k].grid(row=ii+1,column=2+j,sticky='nw')
        self.b=Button(self.top,text='Submit',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg,fg=DEFAULT_SETTINGS.root_bg)
        self.b.grid(row=2,column=3+j,sticky='s')
        self.e=Button(self.top,text='cancel',command=self.cancel,bg=DEFAULT_SETTINGS.root_fg,fg=DEFAULT_SETTINGS.root_bg)
        self.e.grid(row=3,column=3+j,sticky='s')
    def cleanup(self):
        targets_dic={}
        for i,(k,v) in enumerate(self.new_dic.items()):
            print(k,':',v.get())
            if v.get().strip()!='':
                targets_dic[k]=v.get().strip()
                self.df['label_i'].replace(k,v.get().strip(),inplace=True)
                self.df['label_i']=self.df['label_i'].astype(str)
            else:
                self.df=self.df[self.df['label_i']!=k].reset_index().drop('index',axis=1)
                self.df['label_i']=self.df['label_i'].astype(str)
        FIND_REPLACE_TARGETS(targets_dic,self.path_annotations)
        self.value='cleanedup'
        
        self.top.destroy()

    def cancel(self):
        #self.value=str(self.clicked.get().split(':')[0])
        self.value='None Selected'
        self.top.destroy()
    def get_update_background_img(self):
        self.image=Image.open(DEFAULT_SETTINGS.root_background_img)
        self.image=self.image.resize((ROOT_W,ROOT_H),Image.ANTIALIAS)
        self.bg=ImageTk.PhotoImage(self.image)
        self.canvas=tk.Canvas(self.top,width=ROOT_W,height=ROOT_H)
        self.canvas.grid(row=0,column=0,columnspan=DEFAULT_SETTINGS.canvas_columnspan,rowspan=DEFAULT_SETTINGS.canvas_rowspan,sticky='nw')
        self.canvas.create_image(0,0,image=self.bg,anchor='nw')



class MOSAIC:
    def __init__(self,path_JPEGImages,path_Annotations,df_filename=None):
        self.DEFAULT_ENCODING = DEFAULT_SETTINGS.DEFAULT_ENCODING #'utf-8'
        self.XML_EXT = DEFAULT_SETTINGS.XML_EXT #'.xml'
        self.JPG_EXT=DEFAULT_SETTINGS.JPG_EXT #'.jpg'
        self.ENCODE_METHOD = self.DEFAULT_ENCODING
        self.W=DEFAULT_SETTINGS.W #100
        self.H=DEFAULT_SETTINGS.H #100
        self.MOSAIC_NUM=DEFAULT_SETTINGS.MOSAIC_NUM #200
        self.go_to_next=False
        self.close_window=False
        self.DX=int(np.ceil(np.sqrt(self.MOSAIC_NUM)))
        self.DY=int(np.ceil(np.sqrt(self.MOSAIC_NUM)))
        self.FIGSIZE_W=DEFAULT_SETTINGS.FIGSIZE_W #50
        self.FIGSIZE_H=DEFAULT_SETTINGS.FIGSIZE_H #50
        self.FIGSIZE_INCH_W=DEFAULT_SETTINGS.FIGSIZE_INCH_W#8
        self.FIGSIZE_INCH_H=DEFAULT_SETTINGS.FIGSIZE_INCH_H#8
        self.COLOR=DEFAULT_SETTINGS.COLOR #'red'
        self.path_JPEGImages=path_JPEGImages#r'/Volumes/One Touch/Images_gdrive/Drone_Images/Training/JPEGImages'
        self.path_Annotations=path_Annotations#r'/Volumes/One Touch/Images_gdrive/Drone_Images/Training/Annotations'
        self.basePath=self.path_Annotations.replace(os.path.basename(self.path_Annotations),"")
        self.path_Annotations_tofix=os.path.join(self.basePath,'Annotations_tofix')
        self.path_JPEGImages_tofix=os.path.join(self.basePath,'JPEGImages_tofix')
        self.path_JPEGImages_tofix_bbox=os.path.join(self.basePath,'JPEGImages_tofix_bbox')
        self.not_checked=tk.StringVar()
        self.checked_bad=tk.StringVar()
        self.checked_good=tk.StringVar()
        self.total_not_checked=tk.StringVar()
        self.total_checked_good=tk.StringVar()
        self.total_checked_bad=tk.StringVar()
        self.useSSIM=DEFAULT_SETTINGS.useSSIM
        try:
            self.difficult_keyword=DEFAULT_SETTINGS.difficult_keyword
        except:
            self.difficult_keyword='difficult' #used for sorting annotations

        self.plotting_dx_dy=False
        self.ROOT_H=ROOT_H
        self.ROOT_W=ROOT_W
        self.inspect_mosaic=False
        self.close_window_mosaic=False
        self.run_selection=None
        self.ex2=None
        self.ey2=None

        if os.path.exists(self.path_Annotations_tofix)==False:
            os.makedirs(self.path_Annotations_tofix)
        if os.path.exists(self.path_JPEGImages_tofix)==False:
            os.makedirs(self.path_JPEGImages_tofix)
        if os.path.exists(self.path_JPEGImages_tofix_bbox)==False:
            os.makedirs(self.path_JPEGImages_tofix_bbox)
        
        if df_filename==None:
            self.df_filename=os.path.join(self.basePath,"df_{}.pkl".format(os.path.basename(os.path.dirname(self.basePath))))
            self.df_filename_csv=self.df_filename.replace('.pkl','.csv')
        else:
            self.df_filename=df_filename
            self.df_filename_csv=self.df_filename.replace('.pkl','.csv')
    def load(self):
        self.df=pd.read_pickle(self.df_filename)
        self.df_filename_csv=self.df_filename.replace('.pkl','.csv')
        #df=pd.DataFrame(columns=['xmin','xmax','ymin','ymax','width','height','label_i','path_jpeg_i','path_anno_i'],dtype='object')
        self.int_cols=['xmin','xmax','ymin','ymax','width','height']
        for int_col in tqdm(self.int_cols):
            self.df[int_col]=[int(np.floor(float(w))) for w in self.df[int_col]]
            self.df[int_col]=self.df[int_col].astype(int)
        if 'checked' not in self.df.columns:
            self.df['checked']=self.df['path_anno_i'].copy()
            self.df['checked']='' 
        if 'DX' not in self.df.columns:
            self.df['DX']=self.df.xmax-self.df.xmin
        if 'DY' not in self.df.columns:
            self.df['DY']=self.df.ymax-self.df.ymin
        if 'difficulty' not in self.df.columns:
            self.df['difficulty']=self.df['label_i'].copy()
            self.df['difficulty']=['0' for w in self.df['difficulty']]
        self.unique_labels={w:i for i,w in enumerate(self.df['label_i'].unique())}
        self.clear_fix_bad()
        self.DX_MIN=min(self.df.DX)
        self.DX_MAX=max(self.df.DX)
        self.DY_MIN=min(self.df.DY)
        self.DY_MAX=max(self.df.DY)
        print(self.DX_MIN)
        print(self.DX_MAX)
        print(self.DY_MIN)
        print(self.DY_MAX)
        i=0
        self.label_dic={}
        for label in self.df['label_i'].unique():
            if label not in self.label_dic.keys():
                self.label_dic[label]=i
                i+=1
        self.df['label_i_int']=self.df['label_i'].copy()
        self.df['label_i_int']=[self.label_dic[w] for w in self.df['label_i']]
        self.rev_label_dic={v:k for k,v in self.label_dic.items()}
        self.SHOWTABLE_BUTTONS()
        self.df.to_pickle(self.df_filename)
        self.df.to_csv(self.df_filename_csv,index=None)

    def remove_blanks(self):
        unique_annos_notblank=self.df.path_anno_i.unique()
        unique_annos_notblank=[os.path.abspath(os.path.join(self.path_Annotations,os.path.basename(w))) for w in unique_annos_notblank]
        unique_annos_total=os.listdir(self.path_Annotations)
        unique_annos_total=[os.path.abspath(os.path.join(self.path_Annotations,w)) for w in unique_annos_total]

        blank_annos=list(set(unique_annos_total)-set(unique_annos_notblank))
        self.blank_annos=blank_annos


        unique_jpegs_notblank=self.df.path_jpeg_i.unique()
        unique_jpegs_notblank=[os.path.abspath(os.path.join(self.path_JPEGImages,os.path.basename(w))) for w in unique_jpegs_notblank]
        unique_jpegs_total=os.listdir(self.path_JPEGImages)
        unique_jpegs_total=[os.path.abspath(os.path.join(self.path_JPEGImages,w)) for w in unique_jpegs_total]
        blank_jpegs=list(set(unique_jpegs_total)-set(unique_jpegs_notblank))
        self.blank_jpegs=blank_jpegs
        self.popup_remove_blanks()
        if self.w.value=="Yes":
            if len(blank_annos)>0:
                print('removing blank annos')
                for blank in tqdm(blank_annos):
                    
                    os.remove(blank)
            else:
                print('no blank annos')


            if len(blank_jpegs)>0:
                print('removing blank jpegs')
                for blank in tqdm(blank_jpegs):
                    os.remove(blank)
            else:
                print('no blank jpegs')

    def SHOWTABLE_BUTTONS(self):
        self.popup_SHOWTABLE_button=Button(root_tk,text='Show df',command=self.popupWindow_showtable,bg=DEFAULT_SETTINGS.root_fg,fg=DEFAULT_SETTINGS.root_bg)
        self.popup_SHOWTABLE_button.grid(row=8,column=2,sticky='sw')

    def popupWindow_showtable(self):
        try:
            self.top.destroy()
        except:
            pass
        self.top=tk.Toplevel(root_tk)
        self.top.geometry( "{}x{}".format(int(root_tk.winfo_screenwidth()*0.95//1.5),int(root_tk.winfo_screenheight()*0.95//1.5)) )
        self.top.configure(background = 'black')
        self.b=Button(self.top,text='Save',command=self.cleanup_show,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.pack()
        self.c=Button(self.top,text='Close',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.c.pack()
        self.show_table()

    def cleanup_show(self):
        self.app.table.saveAs(self.df_filename_csv)
        print('self.df_filename')
        print(self.df_filename_csv)
        df_i=pd.read_csv(self.df_filename_csv,index_col=None)
        columns=df_i.columns
        drop_columns=[w for w in columns if w.find('Unnamed')!=-1]
        df_i.drop(drop_columns,axis=1,inplace=True)
        df_i.to_pickle(self.df_filename,protocol=2)
        df_i.to_csv(self.df_filename_csv,index=None)
        self.load()
        self.top.destroy()
    def cleanup(self):
        self.top.destroy()
    def show_table(self):
        self.app = TestApp(self.top, self.df_filename_csv)
        self.app.pack(fill=tk.BOTH, expand=1)

    def popup_remove_blanks(self):
        root_tk.title('Do you want to remove these blanks?')
        self.w=popupWindow_REMOVEBLANKS(root_tk,self.blank_annos,self.blank_jpegs)
        root_tk.wait_window(self.w.top)
        print(self.w.value)
        root_tk.title("MOSAIC Chip Sorter")
        return self.w.value

    def popup_create_blackpatches(self):
        root_tk.title('Do you want to create black patches on these classes?')
        self.w=popupWindow_BLACKPATCHES(root_tk,self.df,self.df_filename,self.df_filename_csv)
        root_tk.wait_window(self.w.top)
        print(self.w.value)
        root_tk.title("MOSAIC Chip Sorter")
        return self.w.value

    def popup_create_fakepatches(self):
        root_tk.title('Do you want to create fake patches on these classes to random backgrounds in the FAKE_BACKGROUNDS directory?')
        self.w=popupWindow_FAKEPATCHES(root_tk,self.df,self.df_filename,self.df_filename_csv)
        root_tk.wait_window(self.w.top)
        print(self.w.value)
        root_tk.title("MOSAIC Chip Sorter")
        return self.w.value

    def export_new_df(self):
        self.new_df_basepath=os.path.dirname(self.path_Annotations)
        self.new_df_basepath=os.path.join(self.new_df_basepath,'NEW_DATASET')
        if os.path.exists(self.new_df_basepath)==False:
            os.makedirs(self.new_df_basepath)
        else:
            #TD poupwindow to decide to remove existing
            #os.system(f'rm -rf {self.new_df_basepath}')
            self.new_df_basepath=self.new_df_basepath+"_"+str(time.time()).split('.')[0]
            os.makedirs(self.new_df_basepath)

        self.path_Annotations_new_df=os.path.join(self.new_df_basepath,'Annotations')
        self.path_JPEGImages_new_df=os.path.join(self.new_df_basepath,'JPEGImages')
        if os.path.exists(self.path_Annotations_new_df)==False:
            os.makedirs(self.path_Annotations_new_df)
        else:
            os.system(f'rm -rf {self.path_Annotations_new_df}')
            os.makedirs(self.path_Annotations_new_df)
        if os.path.exists(self.path_JPEGImages_new_df)==False:
            os.makedirs(self.path_JPEGImages_new_df)
        else:
            os.system(f'rm -rf {self.path_JPEGImages_new_df}')
            os.makedirs(self.path_JPEGImages_new_df)
        print('Copying Selection of JPEGImages to NEW_DATASET/JPEGImages')
        for jpg_i in tqdm(self.new_df['path_jpeg_i']):
            shutil.copy(jpg_i,self.path_JPEGImages_new_df)
        self.new_df['path_jpeg_i']=[os.path.join(self.path_JPEGImages_new_df,os.path.basename(w)) for w in self.new_df['path_jpeg_i']]

        print('Copying Selection of Annotations to NEW_DATASET/Annotations')
        for anno_i in tqdm(self.new_df['path_anno_i']):
            shutil.copy(anno_i,self.path_Annotations_new_df)
        self.new_df['path_anno_i']=[os.path.join(self.path_Annotations_new_df,os.path.basename(w)) for w in self.new_df['path_anno_i']]

        self.new_df_filename=os.path.join(self.new_df_basepath,"df_{}.pkl".format(os.path.basename(self.new_df_basepath)))
        self.new_df_filename_csv=self.new_df_filename.replace('.pkl','.csv')       

        self.new_df.to_pickle(self.new_df_filename)
        self.new_df.to_csv(self.new_df_filename_csv,index=None)

        from multiprocessing import Process
        # Start new process with our new dataset
        cmd_i=f'{self.PYTHON_PATH} MOSAIC_Chip_Sorter.py --path_Annotations "{self.path_Annotations_new_df}" --path_JPEGImages "{self.path_JPEGImages_new_df}"'
        Process(target=self.run_cmd,args=(cmd_i,)).start()

    def create_new_df(self):
        self.new_df=self.df.copy()
        self.new_df=self.popup_create_newdf(self.new_df)
        if len(self.new_df)>0:
            self.export_new_df()
        else:
            print('No new dataset is being created')
        #TD popupwindow to decide what classes to keep

    def create_blackpatches(self):
        self.df=self.popup_create_blackpatches()
    
    def create_fakepatches(self):
        self.df=self.popup_create_fakepatches()

    def plot_dx_dy(self):
        if _platform!='darwin':
            plt.clf() #edit sjs 5/28/2022
        self.plotting_dx_dy=True
        self.fig_k=plt.figure(figsize=(self.FIGSIZE_W,self.FIGSIZE_H),num='BBOX SCATTER PLOT of SIZES DX vs. DY.  "Double Click" to inspect.  Press "q" to quit.    Press "m" for MOSAIC.')
        self.fig_k.set_size_inches((self.FIGSIZE_INCH_W, self.FIGSIZE_INCH_W))
        self.cidj = self.fig_k.canvas.mpl_connect('button_press_event', self.onclick_show_dxdy)
        self.cidkj = self.fig_k.canvas.mpl_connect('key_press_event', self.on_key_show_dxdy)
        self.cidkjm = self.fig_k.canvas.mpl_connect('button_release_event',self.release_show_dxdy)
        self.close_window_mosaic=False

        plt.rcParams['axes.facecolor'] = 'gray'
        plt.grid(c='white')
        xaxis_labels='DX (XMAX- XMIN)\nPress "d" to delete annotation for this object.\n'    
        for label_j,int_i in self.label_dic.items():
            xaxis_labels+='Press "{}" to change label to "{}"\n'.format(int_i,label_j)  
        plt.xlabel(xaxis_labels)
        plt.ylabel('DY (YMAX-YMIN)')
        plt.scatter(self.df.DX,self.df.DY,c=self.df['label_i_int'],cmap='Spectral',s=5)
        self.gca=plt.gca()
        self.gca.set_aspect('equal','datalim')
        if len(self.df['label_i_int'].unique())>1:
            plt.colorbar(boundaries=np.arange(len(self.df['label_i_int'].unique())+1)-0.5).set_ticks(np.arange(len(self.df['label_i_int'].unique())))
        plt.tight_layout()
        plt.show()
    def filter_dxdy(self,DX_MIN,DX_MAX,DY_MIN,DY_MAX):
        self.new_df=self.df[(self.df['DX']>=DX_MIN) & (self.df['DX']<=DX_MAX) & (self.df['DY']>=DY_MIN) & (self.df['DY']<=DY_MAX)].copy()#.reset_index().drop('index',axis=1).copy()
        self.bad_index_list=[]
        for row in tqdm(range(len(self.df))):
            self.index_bad=self.df.iloc[row].name
            path_jpeg_i=self.df['path_jpeg_i'].iloc[row]
            list_of_acceptable_jpegs=list(self.new_df['path_jpeg_i'].unique())
            label_k=self.df['label_i'][self.index_bad]
            self.new_df_index_list=list(self.new_df.index)
            if path_jpeg_i not in list_of_acceptable_jpegs:
                if os.path.exists(path_jpeg_i):
                    os.remove(path_jpeg_i)
                self.bad_index_list.append(self.index_bad)
                path_anno_bad=self.df['path_anno_i'][self.index_bad]
                if os.path.exists(path_anno_bad):
                    os.remove(path_anno_bad)
            elif row in self.new_df_index_list:
                pass
            else:
                xmin=self.df['xmin'].iloc[row]
                xmax=self.df['xmax'].iloc[row]
                ymin=self.df['ymin'].iloc[row]
                ymax=self.df['ymax'].iloc[row]

                #try:
                
                xmin=str(xmin)
                ymin=str(ymin)
                xmax=str(xmax)
                ymax=str(ymax)

                
                path_anno_bad=self.df['path_anno_i'][self.index_bad]

                assert path_jpeg_i==self.df['path_jpeg_i'][self.index_bad]
                if os.path.exists(path_anno_bad):
                    f=open(path_anno_bad,'r')
                    f_read=f.readlines()
                    f.close()
                    try:
                        f_read[0].find('annotation')!=-1
                    except:
                        pprint(f_read)
                        assert f_read[0].find('annotation')!=-1
                    f_new=[]
                    start_line=0
                    end_line=0

                    self.bad_index_list.append(self.index_bad)
                    for ii,line in enumerate(f_read):
                        if line.find(label_k)!=-1:   
                            combo_i=f_read[ii-1:]
                            combo_i="".join([w.replace('\n','') for w in combo_i])
                            combo_i=combo_i.split('<object>')
                            #print(len(combo_i),combo_i)
                            #if len(combo_i)>1:
                            if combo_i[1].find(xmin)!=-1 and combo_i[1].find(xmax)!=-1 and combo_i[1].find(ymin)!=-1 and combo_i[1].find(ymax)!=-1:
                                start_line=ii-1
                                for jj,line_j in enumerate(combo_i[1].split('</')):
                                    if line_j.find('object>')!=-1:
                                        end_line=jj+ii
                                        print('found label_k',label_k)
                                        print('deleting bounding box')
                                        print('xmin',xmin)
                                        print('xmax',xmax)
                                        print('ymin',ymin)
                                        print('ymax',ymax)
                                f_new.append(line)
                            else:
                                f_new.append(line)
                            #else:
                            #    f_new.append(line)
                        else:
                            f_new.append(line)
                    if end_line!=0:
                        f_new=f_new[:start_line]+f_new[end_line+1:]

                    try:
                        f_new[0].find('annotation')!=-1
                    except:
                        pprint(f_read)
                        assert f_new[0].find('annotation')!=-1
                    f=open(path_anno_bad,'w')
                    [f.writelines(w) for w in f_new]
                    f.close()

                #except:
                #    print('This ship has sailed, item not found.')
        print('cleaning  up the dataframe')
        for bad_index in tqdm(self.bad_index_list):
            self.df=self.df.drop(bad_index,axis=0)
        self.df=self.df.reset_index().drop('index',axis=1)
        self.df.to_pickle(self.df_filename)
        self.df.to_csv(self.df_filename_csv,index=None)

    def show_draw(self):
        #TBD
        pass


    def draw(self):
        self.df_fix=self.df[self.df['checked']=='bad'].reset_index().drop('index',axis=1)
        self.unique_labels={w:i for i,w in enumerate(self.df['label_i'].unique())}
        for anno,jpg in zip(self.df_fix.path_anno_i,self.df_fix.path_jpeg_i):
            anno_i=os.path.basename(anno) #.split('/')[-1]
            jpg_i=os.path.basename(jpg) #.split('/')[-1]
            path_fix_anno=os.path.join(self.path_Annotations_tofix,anno_i)
            path_fix_jpeg=os.path.join(self.path_JPEGImages_tofix_bbox,jpg_i)
            self.df_to_fix_i=self.df_fix[self.df_fix['path_anno_i']==anno].reset_index().drop('index',axis=1)
            for row in range(len(self.df_to_fix_i)):
                image=Image.open(path_fix_jpeg)
                xmin=self.df_to_fix_i['xmin'].iloc[row]
                xmax=self.df_to_fix_i['xmax'].iloc[row]
                ymin=self.df_to_fix_i['ymin'].iloc[row]
                ymax=self.df_to_fix_i['ymax'].iloc[row]
                label_i=self.df_to_fix_i['label_i'].iloc[row]
                self.draw_objects(ImageDraw.Draw(image),xmin,xmax,ymin,ymax,label_i,path_fix_jpeg,image)
    def pad(self,str_i,min_len=8):
        while len(str_i)<min_len:
            str_i='0'+str_i
        return str_i
    def bb_intersection(self,boxA,boxB):
        xA=max(boxA[0],boxB[0])
        yA=max(boxA[1],boxB[1])
        xB=min(boxA[2],boxB[2])
        yB=min(boxA[3],boxB[3])
        interArea=max(0,xB-xA+1)*max(0,yB-yA+1)
        boxAArea=(boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
        boxBArea=(boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
        denominator=float(boxAArea+boxBArea-interArea)
        if denominator>0:
            iou=interArea/float(boxAArea+boxBArea-interArea)
        else:
            iou=1
        return iou
    def chips(self,BLANK):
        CHIP_LIST=self.popup_chips(self.df['label_i'].unique())
        if len(CHIP_LIST)>0:
            self.unique_labels={w:i for i,w in enumerate(CHIP_LIST)}
            self.basePath_chips=os.path.join(self.basePath,'chips')
            if os.path.exists(self.basePath_chips):
                CLEAR_EXISTING_CHIPS=self.popup_chips_CLEAREXISTING(self.basePath_chips)
                if CLEAR_EXISTING_CHIPS:
                    if os.path.exists(self.basePath_chips):
                        remove_directory(self.basePath_chips)
                #os.system('rm -rf {}'.format(self.basePath_chips))
            try:
                os.makedirs(self.basePath_chips)
            except:
                print('Already exists')
            for label_i in tqdm(self.unique_labels):
                self.df_chips=self.df[self.df['label_i']==label_i].reset_index().drop('index',axis=1).copy()
                self.path_chips_label_i=os.path.join(self.basePath_chips,label_i)
                if BLANK:
                    self.path_chips_label_i_BLANK=os.path.join(self.basePath_chips,label_i+'_BLANK')
                if os.path.exists(self.path_chips_label_i):
                    pass
                else:
                    os.makedirs(self.path_chips_label_i)
                if BLANK:
                    if os.path.exists(self.path_chips_label_i_BLANK):
                        pass
                    else:
                        os.makedirs(self.path_chips_label_i_BLANK)
                for anno,jpg in tqdm(zip(self.df_chips.path_anno_i,self.df_chips.path_jpeg_i)):
                    anno_i=os.path.basename(anno) #.split('/')[-1]
                    jpg_i=os.path.basename(jpg) #.split('/')[-1]
                    path_chip_anno=os.path.join(self.path_Annotations,anno_i)
                    path_chip_jpeg=os.path.join(self.path_JPEGImages,jpg_i)
                    self.df_chips_i=self.df_chips[self.df_chips['path_anno_i']==anno].reset_index().drop('index',axis=1)
                    image=cv2.imread(path_chip_jpeg)
                    for j,row in enumerate(range(len(self.df_chips_i))):
                        xmin=self.df_chips_i['xmin'].iloc[row]
                        xmax=self.df_chips_i['xmax'].iloc[row]
                        ymin=self.df_chips_i['ymin'].iloc[row]
                        ymax=self.df_chips_i['ymax'].iloc[row]
                        label_i=self.df_chips_i['label_i'].iloc[row]
                        if BLANK:
                            label_i_BLANK=label_i+'_BLANK'
                        chipA=image[ymin:ymax,xmin:xmax]
                        chip_name_i=label_i+"_"+jpg_i.split('.')[0]+'_chip{}of{}_'.format(self.pad(str(j)),self.pad(str(len(self.df_chips_i))))+'_ymin{}'.format(ymin)+'_ymax{}'.format(ymax)+'_xmin{}'.format(xmin)+'_xmax{}.jpg'.format(xmax)
                        chip_name_i=os.path.join(self.path_chips_label_i,chip_name_i)
                        try:
                            cv2.imwrite(chip_name_i,chipA)
                            if BLANK:
                                #CREATE BLANK CHIP
                                dx=xmax-xmin
                                factor=1.5
                                if xmin>0.5*image.shape[1]:
                                    xminB=xmin-dx*factor
                                    xmaxB=xmax-dx*factor
                                else:
                                    xminB=xmin+dx*factor
                                    xmaxB=xmax+dx*factor
                                xminB=int(max(0,xminB))
                                xmaxB=int(min(image.shape[1],xmaxB))
                                dy=ymax-ymin
                                if ymin>0.5*image.shape[0]:
                                    yminB=ymin-dy*factor
                                    ymaxB=ymax-dy*factor
                                else:
                                    yminB=ymin+dy*factor
                                    ymaxB=ymax+dy*factor
                                yminB=int(max(0,yminB))
                                ymaxB=int(min(image.shape[0],ymaxB))
                                chipBLANK=image[yminB:ymaxB,xminB:xmaxB]
                                boxBlank=(xminB,yminB,xmaxB,ymaxB)                   
                                iou=0
                                for p,rowp in enumerate(range(len(self.df_chips_i))):
                                    xminp=self.df_chips_i['xmin'].iloc[rowp]
                                    xmaxp=self.df_chips_i['xmax'].iloc[rowp]
                                    yminp=self.df_chips_i['ymin'].iloc[rowp]
                                    ymaxp=self.df_chips_i['ymax'].iloc[rowp]
                                    boxFull=(xminp,yminp,xmaxp,ymaxp)
                                    ioup=self.bb_intersection(boxBlank,boxFull)
                                    if ioup>iou:
                                        iou=ioup
                                if iou<0.2:
                                    chip_name_i=label_i_BLANK+"_"+jpg_i.split('.')[0]+'_chip{}of{}_'.format(self.pad(str(j)),self.pad(str(len(self.df_chips_i))))+'_ymin{}'.format(ymin)+'_ymax{}'.format(ymax)+'_xmin{}'.format(xmin)+'_xmax{}.jpg'.format(xmax)
                                    chip_name_i=os.path.join(self.path_chips_label_i_BLANK,chip_name_i)
                                    #print('yminB:ymaxB,xminB:xmaxB')
                                    #print('{}:{},{}:{}'.format(yminB,ymaxB,xminB,xmaxB))
                                    #print('image.shape')
                                    #print(image.shape)
                                    try:
                                        cv2.imwrite(chip_name_i,chipBLANK)
                                    except:
                                        print('BAD ASSERTION,skipping this chip')
                        except:
                            print('BAD ASSERTION,skipping this chip')
    def popup_create_newdf(self,unique_labels):
        root_tk.title('Which classes & difficulty do you want to create a new dataset with?')
        self.w= popupWindow_KEEPCLASS(root_tk,unique_labels)
        root_tk.wait_window(self.w.top)
        root_tk.wm_state('iconic')#test
        print(self.w.value)
        root_tk.title('MOSAIC Chip Sorter')
        return self.w.value
    def popup_chips(self,unique_labels):
        root_tk.title('Which Chips?')
        self.w= popupWindow_CHIPS(root_tk,unique_labels)
        root_tk.wait_window(self.w.top)
        print(self.w.value)
        root_tk.title('MOSAIC Chip Sorter')
        return self.w.value
    def popup_chips_CLEAREXISTING(self,path_chips):
        root_tk.title('CLEAR EXISTING CHIPS?')
        self.w= popupWindow_CHIPS_CLEAR(root_tk,path_chips)
        root_tk.wait_window(self.w.top)
        print(self.w.value)
        root_tk.title('MOSAIC Chip Sorter')
        return self.w.value
    def popup(self):
        root_tk.title('')
        self.w=popupWindow(root_tk)
        root_tk.wait_window(self.w.top)
        print(self.w.value)
        root_tk.title("MOSAIC Chip Sorter")
        return self.w.value

    def popup_multint(self,options):
        if _platform=='darwin':
            root_tk.withdraw()
        root_tk.title('Select the new object name from the dropdown')
        original_root_tk=root_tk
        self.w=popupWindowDropDown(root_tk,options)
        original_root_tk.wait_window(self.w.top)
        print(self.w.value)
        root_tk.title("MOSAIC Chip Sorter")
        if _platform=='darwin':
            original_root_tk.update()
            original_root_tk.deiconify()
        return self.w.value

    def popup_changelabels(self):
        #if _platform=='darwin':
        root_tk.withdraw()
        root_tk.title('Change Labels')
        original_root_tk=root_tk
        print(self.label_dic)
        self.w=popupWindowChangeLabels(root_tk,self.label_dic,self.path_Annotations,self.df)
        original_root_tk.wait_window(self.w.top)
        print(self.w.value)
        root_tk.title("MOSAIC Chip Sorter")
        #if _platform=='darwin':
        original_root_tk.update()
        original_root_tk.deiconify()
        self.df=self.w.df
        self.df.to_pickle(self.df_filename)
        self.df.to_csv(self.df_filename_csv,index=None)
        return self.w.value
    def draw_objects(self,draw, xmin,xmax,ymin,ymax,label_i,path_fix_jpeg,image):
        """Draws the bounding box and label for each object."""
        
        draw.rectangle([(xmin, ymin),
                        (xmax, ymax)],
                    outline=self.COLOR, width=3)
        font = ImageFont.load_default()
        draw.text((xmin + 4, ymin + 4),
                '%s\n' % (label_i),
                fill=self.COLOR, font=font)
        image.save(path_fix_jpeg)
    def clear_checked(self):
        self.df['checked']=''
        self.df.to_pickle(self.df_filename)
    def move_fix(self):
        self.df_fix=self.df[self.df['checked']=='bad'].reset_index().drop('index',axis=1)
        for anno,jpg in zip(self.df_fix.path_anno_i,self.df_fix.path_jpeg_i):
            anno_i=os.path.basename(anno) #.split('/')[-1]
            jpg_i=os.path.basename(jpg) #.split('/')[-1]
            if os.path.exists(os.path.join(self.path_Annotations_tofix,anno_i))==False:
                shutil.copy(anno,self.path_Annotations_tofix)
            if os.path.exists(os.path.join(self.path_JPEGImages_tofix,jpg_i))==False:
                shutil.copy(jpg,self.path_JPEGImages_tofix)
            if os.path.exists(os.path.join(self.path_JPEGImages_tofix_bbox,jpg_i))==False:
                shutil.copy(jpg,self.path_JPEGImages_tofix_bbox)
        print('Annotations to fix = ',len(list(os.listdir(self.path_Annotations_tofix))))
        print('JPEGs to fix = ',len(list(os.listdir(self.path_JPEGImages_tofix))))
        self.draw()

    def merge_fix(self):
        self.df_fix=self.df[self.df['checked']=='bad'].reset_index().drop('index',axis=1)
        for anno,jpg in zip(self.df_fix.path_anno_i,self.df_fix.path_jpeg_i):
            anno_i=os.path.basename(anno) #.split('/')[-1]
            jpg_i=os.path.basename(jpg)#.split('/')[-1]
            if os.path.exists(os.path.join(self.path_Annotations_tofix,anno_i))==True:
                shutil.copy(os.path.join(self.path_Annotations_tofix,anno_i),anno)
        self.df=self.df[self.df['checked']!='bad'].reset_index().drop('index',axis=1)
        self.merge_df()
        self.clear_fix()
        self.clear_checked()
    def update_fix(self):
        print('Annotations to fix = ',len(list(os.listdir(self.path_Annotations_tofix))))
        print('JPEGs to fix = ',len(list(os.listdir(self.path_JPEGImages_tofix))))
        print('JPEGs to fix bbox= ',len(list(os.listdir(self.path_JPEGImages_tofix_bbox))))
        self.annos_to_fix=os.listdir(self.path_Annotations_tofix)
        self.jpegs_to_fix=os.listdir(self.path_JPEGImages_tofix)
        self.jpegs_to_fix_bbox=os.listdir(self.path_JPEGImages_tofix_bbox)         
    def clear_fix(self):
        print('Annotations to fix = ',len(list(os.listdir(self.path_Annotations_tofix))))
        print('JPEGs to fix = ',len(list(os.listdir(self.path_JPEGImages_tofix))))
        annos_to_fix=os.listdir(self.path_Annotations_tofix)
        if len(annos_to_fix)!=0:
            removed_all=[os.remove(os.path.join(self.path_Annotations_tofix,w)) for w in annos_to_fix if w[0]!='.']
        jpegs_to_fix=os.listdir(self.path_JPEGImages_tofix)
        if len(jpegs_to_fix)!=0:
            removed_all=[os.remove(os.path.join(self.path_JPEGImages_tofix,w)) for w in jpegs_to_fix if w[0]!='.']
        jpegs_to_fix_bbox=os.listdir(self.path_JPEGImages_tofix_bbox)
        if len(jpegs_to_fix_bbox)!=0:
            removed_all=[os.remove(os.path.join(self.path_JPEGImages_tofix_bbox,w)) for w in jpegs_to_fix_bbox if w[0]!='.']
        print('Annotations to fix = ',len(list(os.listdir(self.path_Annotations_tofix))))
        print('JPEGs to fix = ',len(list(os.listdir(self.path_JPEGImages_tofix))))
    def clear_fix_bad(self):
        annos_to_fix=os.listdir(self.path_Annotations_tofix)
        if len(annos_to_fix)!=0:
            removed_all=[os.remove(os.path.join(self.path_Annotations_tofix,w)) for w in annos_to_fix if w[0]=='.']
        jpegs_to_fix=os.listdir(self.path_JPEGImages_tofix)
        if len(jpegs_to_fix)!=0:
            removed_all=[os.remove(os.path.join(self.path_JPEGImages_tofix,w)) for w in jpegs_to_fix if w[0]=='.']
        jpegs_to_fix_bbox=os.listdir(self.path_JPEGImages_tofix_bbox)
        if len(jpegs_to_fix_bbox)!=0:
            removed_all=[os.remove(os.path.join(self.path_JPEGImages_tofix_bbox,w)) for w in jpegs_to_fix_bbox if w[0]=='.']
    def merge_df(self):
        self.Annotations_list_tofix=list(os.listdir(self.path_Annotations_tofix))
        self.Annotations_tofix=[os.path.join(self.path_Annotations_tofix,Anno) for Anno in self.Annotations_list_tofix if Anno.find('.xml')!=-1]
        self.JPEGs_list_tofix=list(os.listdir(self.path_JPEGImages_tofix))
        self.JPEGs_tofix=[os.path.join(self.path_JPEGImages,Anno.split(self.XML_EXT)[0]+self.JPG_EXT) for Anno in self.Annotations_list_tofix if Anno.split(self.XML_EXT)[0]+self.JPG_EXT in self.JPEGs_list_tofix]
        assert len(self.JPEGs_tofix)==len(self.Annotations_tofix) 

        #self.df=pd.DataFrame(columns=['xmin','xmax','ymin','ymax','width','height','label_i','path_jpeg_i','path_anno_i'],dtype='object')
        i=len(self.df)
        for path_anno_i,path_jpeg_i in tqdm(zip(self.Annotations_tofix,self.JPEGs_tofix)):
            parser = etree.XMLParser(encoding=self.ENCODE_METHOD)
            xmltree = ElementTree.parse(path_anno_i, parser=parser).getroot()
            filename = xmltree.find('filename').text
            width_i=xmltree.find('size').find('width').text
            height_i=xmltree.find('size').find('height').text

            for object_iter in xmltree.findall('object'):
                bndbox = object_iter.find("bndbox")
                label = object_iter.find('name').text
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text
                try:
                    difficulty_i=object_iter.find(self.difficult_keyword).text
                except:
                    difficulty_i='0'
                self.df.at[i,'xmin']=xmin
                self.df.at[i,'xmax']=xmax
                self.df.at[i,'ymin']=ymin
                self.df.at[i,'ymax']=ymax
                self.df.at[i,'width']=width_i
                self.df.at[i,'height']=height_i
                self.df.at[i,'label_i']=label
                self.df.at[i,'path_jpeg_i']=path_jpeg_i.replace('_tofix','')
                self.df.at[i,'path_anno_i']=path_anno_i.replace('_tofix','')
                self.df.at[i,'difficulty']=difficulty_i
                self.df.at[i,'checked']=''
                i+=1
        self.df.to_pickle(self.df_filename)
        self.df.to_csv(self.df_filename_csv,index=None)
    def speed_df(self,path_anno_i,path_jpeg_i,df_queue_i):
        df=df_queue_i.get()
        i=0
        parser = etree.XMLParser(encoding=self.ENCODE_METHOD)
        try:
            xmltree = ElementTree.parse(path_anno_i, parser=parser).getroot()
        except:
            print(path_anno_i)
            f=open(path_anno_i,'r')
            f_read=f.readlines()
            f.close()
            f_new=[]
            if f_read[0].find('annotation')==-1:
                f_new.append('<annotation>\n')
                for line in f_read:
                    f_new.append(line)
                f=open(path_anno_i,'w')
                [f.writelines(w) for w in f_new]
                f.close()
            xmltree = ElementTree.parse(path_anno_i, parser=parser).getroot()
        filename = xmltree.find('filename').text
        width_i=xmltree.find('size').find('width').text
        height_i=xmltree.find('size').find('height').text

        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            label = object_iter.find('name').text
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text
            try:
                difficulty_i=object_iter.find(self.difficult_keyword).text
            except:
                difficulty_i='0'
            df.at[i,'xmin']=xmin
            df.at[i,'xmax']=xmax
            df.at[i,'ymin']=ymin
            df.at[i,'ymax']=ymax
            df.at[i,'width']=width_i
            df.at[i,'height']=height_i
            df.at[i,'label_i']=label
            df.at[i,'path_jpeg_i']=path_jpeg_i
            df.at[i,'path_anno_i']=path_anno_i
            df.at[i,'difficulty']=difficulty_i
            i+=1    
        df_queue_i.put(df)    

    def create_df(self):
        self.Annotations_list=list(os.listdir(self.path_Annotations))
        self.JPEGs_list=list(os.listdir(self.path_JPEGImages))
        self.Annotations=[os.path.join(self.path_Annotations,Anno) for Anno in self.Annotations_list if Anno.find(self.XML_EXT)!=-1 and Anno.replace(self.XML_EXT,self.JPG_EXT) in self.JPEGs_list]
        self.JPEGs=[os.path.join(self.path_JPEGImages,Anno.split(self.XML_EXT)[0]+self.JPG_EXT) for Anno in self.Annotations_list if Anno.split(self.XML_EXT)[0]+self.JPG_EXT in self.JPEGs_list]
        assert len(self.JPEGs)==len(self.Annotations) 

        self.df=pd.DataFrame(columns=['xmin','xmax','ymin','ymax','width','height','label_i','path_jpeg_i','path_anno_i','difficulty'],dtype='object')

        from multiprocessing import Process,Queue

        if multiprocessing.cpu_count()>1:
            NUM_PROCESS=multiprocessing.cpu_count()-1
        else:
            NUM_PROCESS=1
        i=0
        processes=[]
        df_queues={}
        for j,(path_anno_i,path_jpeg_i) in tqdm(enumerate(zip(self.Annotations,self.JPEGs))):
            df_queues[len(processes)]=Queue()
            df_queues[len(processes)].put(self.df.copy())
            p=Process(target=self.speed_df,args=(path_anno_i,path_jpeg_i,df_queues[len(processes)]))
            processes.append(p)
            p.start()
            if (j%NUM_PROCESS==0 and j!=0):
                print('\Finished Reading {} Annotations of {} \n'.format(j,len(self.Annotations)))
                for process_i in processes:
                    process_i.join()
                for queue_i in df_queues.values():
                    if i==0:
                        self.df_tmp=queue_i.get()
                        i+=1
                    else:
                        self.df_tmp=pd.concat([self.df_tmp,queue_i.get()],ignore_index=True)
                df_queues={}
                processes=[]
        try:
            for process_i in processes:
                process_i.join()
            for queue_i in df_queues.values():
                if i==0:
                    self.df_tmp=queue_i.get()
                    i+=1
                else:
                    self.df_tmp=pd.concat([self.df_tmp,queue_i.get()],ignore_index=True)
        except:
            pass
        self.df=self.df_tmp.copy()
        self.df.to_pickle(self.df_filename)
        self.df.to_csv(self.df_filename_csv,index=None)
    # def create_df(self):
    #     self.Annotations_list=list(os.listdir(self.path_Annotations))
    #     self.JPEGs_list=list(os.listdir(self.path_JPEGImages))
    #     self.Annotations=[os.path.join(self.path_Annotations,Anno) for Anno in self.Annotations_list if Anno.find(self.XML_EXT)!=-1 and Anno.replace(self.XML_EXT,self.JPG_EXT) in self.JPEGs_list]
    #     self.JPEGs=[os.path.join(self.path_JPEGImages,Anno.split(self.XML_EXT)[0]+self.JPG_EXT) for Anno in self.Annotations_list if Anno.split(self.XML_EXT)[0]+self.JPG_EXT in self.JPEGs_list]
    #     assert len(self.JPEGs)==len(self.Annotations) 

    #     self.df=pd.DataFrame(columns=['xmin','xmax','ymin','ymax','width','height','label_i','path_jpeg_i','path_anno_i'],dtype='object')
    #     i=0
    #     for path_anno_i,path_jpeg_i in tqdm(zip(self.Annotations,self.JPEGs)):
    #         parser = etree.XMLParser(encoding=self.ENCODE_METHOD)
    #         try:
    #             xmltree = ElementTree.parse(path_anno_i, parser=parser).getroot()
    #         except:
    #             print(path_anno_i)
    #             xmltree = ElementTree.parse(path_anno_i, parser=parser).getroot()
    #         filename = xmltree.find('filename').text
    #         width_i=xmltree.find('size').find('width').text
    #         height_i=xmltree.find('size').find('height').text

    #         for object_iter in xmltree.findall('object'):
    #             bndbox = object_iter.find("bndbox")
    #             label = object_iter.find('name').text
    #             xmin = bndbox.find('xmin').text
    #             ymin = bndbox.find('ymin').text
    #             xmax = bndbox.find('xmax').text
    #             ymax = bndbox.find('ymax').text
    #             self.df.at[i,'xmin']=xmin
    #             self.df.at[i,'xmax']=xmax
    #             self.df.at[i,'ymin']=ymin
    #             self.df.at[i,'ymax']=ymax
    #             self.df.at[i,'width']=width_i
    #             self.df.at[i,'height']=height_i
    #             self.df.at[i,'label_i']=label
    #             self.df.at[i,'path_jpeg_i']=path_jpeg_i
    #             self.df.at[i,'path_anno_i']=path_anno_i
    #             i+=1
    #     self.df.to_pickle(self.df_filename)
    #     self.df.to_csv(self.df_filename_csv,index=None)
    def update_checked(self):
        not_checked=str(len(self.df[(self.df['checked']=="")&(self.df['label_i']==self.target_i)]))
        #print('not checked_total = ',not_checked_total)
        self.not_checked.set(not_checked)

        checked_bad=str(len(self.df[(self.df['checked']=='bad')&(self.df['label_i']==self.target_i)]))
        #print('checked_bad = ',checked_bad)
        self.checked_bad.set(checked_bad)

        checked_good=str(len(self.df[(self.df['checked']=="good") &(self.df['label_i']==self.target_i)]))
        #print('checked_good = ',checked_good)
        self.checked_good.set(checked_good)

        total_not_checked=str(len(self.df[(self.df['checked']=="")]))
        #print('total not checked_total = ',total_not_checked_total)
        self.total_not_checked.set(total_not_checked)

        total_checked_bad=str(len(self.df[(self.df['checked']=='bad')]))
        #print('checked_bad = ',total_checked_bad)
        self.total_checked_bad.set(total_checked_bad)

        total_checked_good=str(len(self.df[(self.df['checked']=="good")]))
        #print('total_checked_good = ',total_checked_good)
        self.total_checked_good.set(total_checked_good)

    def on_key(self,event):
        print('you pressed', event.key, event.xdata, event.ydata)
        if event.key=='n':
            plt.close('all')
            cv2.destroyAllWindows() #edit sjs 6/3/2022
            self.go_to_next=True
        if event.key=='q' or event.key=='escape':
            plt.close('all') 
            cv2.destroyAllWindows() #edit sjs 6/3/2022
            self.close_window=True
        self.df.to_pickle(self.df_filename)
        self.df.to_csv(self.df_filename_csv,index=None)
    def on_key_object(self,event):
        print('you pressed', event.key, event.xdata, event.ydata)
        if event.key=='n':
            plt.close('all')
            cv2.destroyAllWindows() #edit sjs 6/3/2022
            self.go_to_next=True
        if event.key=='q' or event.key=='escape':
            plt.close('all') 
            cv2.destroyAllWindows() #edit sjs 6/3/2022
            self.close_window=True
        if event.key=='f':
            for index_i in self.index_i_fix:
                self.df.at[index_i,'checked']="bad"
        if event.key=='u':
            for index_i in self.index_i_fix:
                self.df.at[index_i,'checked']="good"  
        if event.key=='f' or event.key=='u':              
            self.df.to_pickle(self.df_filename)
            self.df.to_csv(self.df_filename_csv,index=None)
    def onclick(self,event):
        #global df_i,axes_list,df,title_list,img_list
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
             ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        print('event.inaxes',event.inaxes)
        for j,ax_j in enumerate(self.axes_list):
            if ax_j==event.inaxes:
                print('subplot j=',j)
                index_bad=self.df_i.iloc[self.dic[j]].name
                if event.dblclick:
                    self.df.at[index_bad,'checked']="good" #resets
                    self.title_list[j].set_text(str(self.dic[j]))
                elif event.button==2:
                    cv2.destroyAllWindows()
                    self.image=Image.open(self.df['path_jpeg_i'].loc[index_bad])
                    xmin=self.df['xmin'].loc[index_bad]
                    xmax=self.df['xmax'].loc[index_bad]
                    ymin=self.df['ymin'].loc[index_bad]
                    ymax=self.df['ymax'].loc[index_bad]
                    label_i=self.df['label_i'].loc[index_bad]
                    draw=ImageDraw.Draw(self.image)
                    draw.rectangle([(xmin, ymin),
                                (xmax, ymax)],
                            outline=self.COLOR, width=3)
                    font = ImageFont.load_default()
                    draw.text((xmin + 4, ymin + 4),
                        '%s\n' % (label_i),
                        fill=self.COLOR, font=font)
                    self.img_i=cv2.cvtColor(np.array(self.image),cv2.COLOR_BGR2RGB)
                    self.img_i_H=self.img_i.shape[1]
                    self.img_i_W=self.img_i.shape[0]
                    self.img_i_W_ratio=self.ROOT_W/self.img_i_W
                    self.img_i_H_ratio=self.ROOT_H/self.img_i_H
                    self.img_i_new_W=int(0.85*self.img_i_W_ratio*self.img_i_W)
                    self.img_i_new_H=int(0.85*self.img_i_H_ratio*self.img_i_H)
                    self.img_i=cv2.resize(self.img_i,(self.img_i_new_W,self.img_i_new_H))
                    cv2.imshow('Selected Image.  Press "q" to close window',self.img_i)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    plt.show() 
                else:
                    self.df.at[index_bad,'checked']="bad" #resets
                    self.title_list[j].set_text('{} = BAD'.format(self.dic[j]))
                print(self.df.loc[index_bad])
                plt.show()
                #plt.pause(1e-3)
                break
        self.df.to_pickle(self.df_filename)
        self.df.to_csv(self.df_filename_csv,index=None)

    def onclick_select_mosaic(self,event):
        #global df_i,axes_list,df,title_list,img_list
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
             ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        print('event.inaxes',event.inaxes)
        for j,ax_j in enumerate(self.axes_list):
            if ax_j==event.inaxes:
                print('subplot j=',j)
                index_bad=self.df_i.iloc[self.dic[j]].name
                
                if event.dblclick:
                    self.df_sample.at[index_bad,'selected']=True #selected is true
                    self.title_list[j].set_text('{}'.format(self.df['label_i'][index_bad]))
                    self.title_list[j].set_color('blue')
                    if index_bad in self.selection_list.values():
                        self.selection_list.pop(j)
                elif event.button==2:
                    cv2.destroyAllWindows()
                    self.image=Image.open(self.df_sample['path_jpeg_i'].loc[index_bad])
                    xmin=self.df_sample['xmin'].loc[index_bad]
                    xmax=self.df_sample['xmax'].loc[index_bad]
                    ymin=self.df_sample['ymin'].loc[index_bad]
                    ymax=self.df_sample['ymax'].loc[index_bad]
                    label_i=self.df_sample['label_i'].loc[index_bad]
                    draw=ImageDraw.Draw(self.image)
                    draw.rectangle([(xmin, ymin),
                                (xmax, ymax)],
                            outline=self.COLOR, width=3)
                    font = ImageFont.load_default()
                    draw.text((xmin + 4, ymin + 4),
                        '%s\n' % (label_i),
                        fill=self.COLOR, font=font)
                    self.img_i=cv2.cvtColor(np.array(self.image),cv2.COLOR_BGR2RGB)
                    self.img_i_H=self.img_i.shape[1]
                    self.img_i_W=self.img_i.shape[0]
                    self.img_i_W_ratio=self.ROOT_W/self.img_i_W
                    self.img_i_H_ratio=self.ROOT_H/self.img_i_H
                    self.img_i_new_W=int(0.85*self.img_i_W_ratio*self.img_i_W)
                    self.img_i_new_H=int(0.85*self.img_i_H_ratio*self.img_i_H)
                    self.img_i=cv2.resize(self.img_i,(self.img_i_new_W,self.img_i_new_H))
                    cv2.imshow('Selected Image.  Press "q" to close window',self.img_i)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    plt.show()                    
                else:
                    self.df_sample.at[index_bad,'selected']=False #selected is false
                    self.title_list[j].set_text('{} = SELECTED'.format(self.dic[j]))
                    self.title_list[j].set_color('red')
                    self.selection_list[j]=index_bad
    
                print(self.df_sample.loc[index_bad])
                plt.show()
                break
        #self.df.to_pickle(self.df_filename) #TBD
    def on_key_mosaic(self,event):
        print('you pressed', event.key, event.xdata, event.ydata)
        if event.key=='q':
            if _platform=='darwin':
                plt.close('all') #edit sjs 5/28
            self.inspect_mosaic=False
            self.run_selection=None
            self.close_window_mosaic=True
            if _platform=='darwin':
                cv2.destroyAllWindows()
            if self.plotting_dx_dy==False:
                self.draw_umap()
            else:
                self.plot_dx_dy()



            #self.close_window_mosaic=True
        elif event.key=='y':
            print('SELECTING ALL')
            self.selection_list={}
            for j,ax_j in enumerate(self.axes_list):
                index_bad=self.df_i.iloc[self.dic[j]].name
                self.df_sample.at[index_bad,'selected']=False #selected is false
                self.title_list[j].set_text('{} = SELECTED'.format(self.dic[j]))
                self.title_list[j].set_color('red')
                self.selection_list[j]=index_bad
            plt.show()
        elif event.key=='u':
            print('DESELECTING ALL')
            self.selection_list={}
            for j,ax_j in enumerate(self.axes_list):
                index_bad=self.df_i.iloc[self.dic[j]].name
                self.df_sample.at[index_bad,'selected']=True #selected is true
                self.title_list[j].set_text('{}'.format(self.df['label_i'][index_bad]))
                self.title_list[j].set_color('purple')
                if index_bad in self.selection_list.values():
                    self.selection_list.pop(j) 
            plt.show()  
        else:
            #plt.close('all')
            if event.key=='n':
                plt.close('all')
                self.go_to_next=True
                self.selection_list={}
            if event.key=='t':
                event.key=self.popup_multint(self.label_dic)
                print('NEW event.key=={}'.format(event.key))           
            if event.key=='d':
                if len(self.selection_list)>0:
                    for j,selection_i in self.selection_list.items():
                        self.index_bad=selection_i
                        try:
                            print('deleting bounding box')
                            xmin=str(self.df['xmin'][self.index_bad])
                            xmax=str(self.df['xmax'][self.index_bad])
                            ymin=str(self.df['ymin'][self.index_bad])
                            ymax=str(self.df['ymax'][self.index_bad])
                            print('xmin',xmin)
                            print('xmax',xmax)
                            print('ymin',ymin)
                            print('ymax',ymax)
                            label_k=self.df['label_i'][self.index_bad]
                            path_anno_bad=self.df['path_anno_i'][self.index_bad]
                            f=open(path_anno_bad,'r')
                            f_read=f.readlines()
                            f.close()
                            f_new=[]
                            start_line=0
                            end_line=0
                            self.df.drop(self.index_bad,axis=0)
                            self.df=self.df.drop(self.index_bad,axis=0)
                            # for ii,line in enumerate(f_read):
                            #     if line.find(label_k)!=-1:
                            #         print('found label_k',label_k)
                            #         combo_i=f_read[ii-1:]
                            #         combo_i="".join([w for w in combo_i])
                            #         combo_i=combo_i.split('<object>')
                            #         if combo_i[1].find(xmin)!=-1 and combo_i[1].find(xmax)!=-1 and combo_i[1].find(ymin)!=-1 and combo_i[1].find(ymax)!=-1:
                            #             start_line=ii-1
                            #             for jj,line_j in enumerate(f_read[ii:]):
                            #                 if line_j.find('</object>')!=-1:
                            #                     end_line=jj+ii
                            #             f_new.append(line)
                            #         else:
                            #             f_new.append(line)
                            #     else:
                            #         f_new.append(line)
                            # f_new=f_new[:start_line]+f_new[end_line+1:]
                            for ii,line in enumerate(f_read):
                                if line.find(label_k)!=-1:   
                                    combo_i=f_read[ii-1:]
                                    combo_i="".join([w.replace('\n','') for w in combo_i])
                                    combo_i=combo_i.split('<object>')
                                    #print(len(combo_i),combo_i)
                                    #if len(combo_i)>1:
                                    if combo_i[1].find(xmin)!=-1 and combo_i[1].find(xmax)!=-1 and combo_i[1].find(ymin)!=-1 and combo_i[1].find(ymax)!=-1:
                                        start_line=ii-1
                                        for jj,line_j in enumerate(combo_i[1].split('</')):
                                            if line_j.find('object>')!=-1:
                                                end_line=jj+ii
                                                print('found label_k',label_k)
                                                print('deleting bounding box')
                                                print('xmin',xmin)
                                                print('xmax',xmax)
                                                print('ymin',ymin)
                                                print('ymax',ymax)
                                        f_new.append(line)
                                    else:
                                        f_new.append(line)
                                    #else:
                                    #    f_new.append(line)
                                else:
                                    f_new.append(line)
                            if end_line!=0:
                                f_new=f_new[:start_line]+f_new[end_line+1:]

                            try:
                                f_new[0].find('annotation')!=-1
                            except:
                                pprint(f_read)
                                assert f_new[0].find('annotation')!=-1
                            f=open(path_anno_bad,'w')
                            [f.writelines(w) for w in f_new]
                            f.close()  
                            self.df.to_pickle(self.df_filename)
                            self.df.to_csv(self.df_filename_csv,index=None)
                        except:
                            print('This ship has sailed, item not found.')
                        self.title_list[j].set_text('{} = DELETED'.format(self.dic[j]))
                        self.title_list[j].set_color('black')
                    plt.show()
                self.selection_list={}
            int_values=[str(w) for w in self.label_dic.values()]
            if len(self.selection_list)>0 and event.key in int_values:
                for j,selection_i in self.selection_list.items():
                    self.index_bad=selection_i
                    for label_j,int_i in self.label_dic.items(): 
                        #print('label_j',label_j,'int_i',int_i)
                        if event.key==str(int_i):
                            print('fixing label')
                            xmin=str(self.df['xmin'][self.index_bad])
                            xmax=str(self.df['xmax'][self.index_bad])
                            ymin=str(self.df['ymin'][self.index_bad])
                            ymax=str(self.df['ymax'][self.index_bad])
                            print('xmin',xmin)
                            print('xmax',xmax)
                            print('ymin',ymin)
                            print('ymax',ymax)
                            label_k=self.df['label_i'][self.index_bad]
                            self.df.at[self.index_bad,'label_i']=self.rev_label_dic[int_i]
                            self.df.at[self.index_bad,'label_i_int']=int_i
                            f=open(self.df['path_anno_i'][self.index_bad],'r')
                            f_read=f.readlines()
                            f.close()
                            f_new=[]
                            for ii,line in enumerate(f_read):
                                if line.find(label_k)!=-1:
                                    print('found label_k',label_k)
                                    combo_i=f_read[ii-1:]
                                    combo_i="".join([w for w in combo_i])
                                    combo_i=combo_i.split('<object>')
                                    #pprint(combo_i)
                                    if combo_i[1].find(xmin)!=-1 and combo_i[1].find(xmax)!=-1 and combo_i[1].find(ymin)!=-1 and combo_i[1].find(ymax)!=-1:
                                        print('fixing it')
                                        print(line.replace(label_k,label_j))
                                        f_new.append(line.replace(label_k,label_j))
                                    else:
                                        f_new.append(line)
                                else:
                                    f_new.append(line)
                            f=open(self.df['path_anno_i'][self.index_bad],'w')
                            [f.writelines(w) for w in f_new]
                            f.close()  
                            self.df.to_pickle(self.df_filename)
                            self.df.to_csv(self.df_filename_csv,index=None)
                            self.title_list[j].set_text('{}'.format(self.df['label_i'][self.index_bad]))
                            self.title_list[j].set_color('green')
                            break    
                            plt.show()
                self.selection_list={}
            #self.df.to_pickle(self.df_filename) #TBD
    def release_show(self,event):
        #global df_i,axes_list,df,title_list,img_list
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
             ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        if event.dblclick==False and self.inspect_mosaic==False:
            #print('event.inaxes',event.inaxes)   
            #print('x1:',self.ex)
            #print('y1:',self.ey)
            self.ex2=event.xdata
            self.ey2=event.ydata
            #print('x2:',self.ex2)
            #print('y2:',self.ey2) 
            if 'umap_input' in self.df.columns:
                self.df_sample=self.df.drop(['umap_input'],axis=1).copy()
            else:
                self.df_sample=self.df.copy()
            self.eymin=min(self.ey2,self.ey)
            self.eymax=max(self.ey2,self.ey)
            self.exmin=min(self.ex2,self.ex)
            self.exmax=max(self.ex2,self.ex)
            self.df_sample=self.df_sample[(self.df_sample['emb_X']>self.exmin) & (self.df_sample['emb_X']<self.exmax) & (self.df_sample['emb_Y']>self.eymin) & (self.df_sample['emb_Y']<self.eymax)]
            print('Total number in this region is: {} \n'.format(len(self.df_sample)))
            self.a_xmin, self.a_xmax, self.a_ymin, self.a_ymax = plt.axis()
            print(self.a_xmin, self.a_xmax, self.a_ymin, self.a_ymax)
    def release_show_dxdy(self,event):
        #global df_i,axes_list,df,title_list,img_list
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
             ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        if event.dblclick==False and self.inspect_mosaic==False:
            #print('event.inaxes',event.inaxes)   
            #print('x1:',self.ex)
            #print('y1:',self.ey)
            self.ex2=event.xdata
            self.ey2=event.ydata
            #print('x2:',self.ex2)
            #print('y2:',self.ey2) 
            if 'umap_input' in self.df.columns:
                self.df_sample=self.df.drop(['umap_input'],axis=1).copy()
            else:
                self.df_sample=self.df.copy()
            self.eymin=min(self.ey2,self.ey)
            self.eymax=max(self.ey2,self.ey)
            self.exmin=min(self.ex2,self.ex)
            self.exmax=max(self.ex2,self.ex)
            self.df_sample=self.df_sample[(self.df_sample['DX']>self.exmin) & (self.df_sample['DX']<self.exmax) & (self.df_sample['DY']>self.eymin) & (self.df_sample['DY']<self.eymax)]
            print('Total number in this region is: {} \n'.format(len(self.df_sample)))
            self.a_xmin, self.a_xmax, self.a_ymin, self.a_ymax = plt.axis()
            print(self.a_xmin, self.a_xmax, self.a_ymin, self.a_ymax)


        #self.df.at[i,'distance']=np.sqrt(self.dx**2+self.dy**2)
        #self.df_to_fix_i=self.df.sort_values(by='distance',ascending=True).copy()
    def onclick_show_dxdy(self,event):
        #global df_i,axes_list,df,title_list,img_list
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
             ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        print('event.inaxes',event.inaxes)
        try:
            #cv2.destroyWindow('Selected Image.  Press "f" to fix.  Press "q" to quit.')
            cv2.destroyAllWindows()
        except:
            pass
        if event.button:
            self.ex=event.xdata
            self.ey=event.ydata
        if event.dblclick:
            self.ex=event.xdata
            self.ey=event.ydata
            if 'umap_input' in self.df.columns:
                self.df_dist=self.df.drop(['umap_input'],axis=1).copy()
            else:
                self.df_dist=self.df.copy()
            self.df_dist['distance']=self.df_dist['xmin'].copy()
            self.df_dist['distance']=0.0
            self.df_dist['distance']=self.df_dist['distance'].astype(np.float16)
            self.df_dist=self.df_dist[(abs(self.df_dist['DX']-self.ex)<2) & (abs(self.df_dist['DY']-self.ey)<2)]
            self.df_dist['dx']=self.df_dist['DX'].copy()
            self.df_dist['dy']=self.df_dist['DY'].copy()
            #for i,(x,y) in enumerate(zip(self.df['emb_X'],self.df['emb_Y'])):
            
            self.df_dist['dx']=self.ex-self.df_dist['dx']
            self.df_dist['dy']=self.ey-self.df_dist['dy']
            self.df_dist['distance']=np.sqrt(self.df_dist['dx']**2+self.df_dist['dy']**2)
            self.df_dist=self.df_dist.sort_values(by='distance',ascending=True)
            #self.df.at[i,'distance']=np.sqrt(self.dx**2+self.dy**2)
            #self.df_to_fix_i=self.df.sort_values(by='distance',ascending=True).copy()
            self.index_bad=self.df_dist.iloc[0].name
            print(self.df_dist.head())
            self.image=Image.open(self.df_dist['path_jpeg_i'].iloc[0])
            xmin=self.df_dist['xmin'].iloc[0]
            xmax=self.df_dist['xmax'].iloc[0]
            ymin=self.df_dist['ymin'].iloc[0]
            ymax=self.df_dist['ymax'].iloc[0]
            label_i=self.df_dist['label_i'].iloc[0]
            draw=ImageDraw.Draw(self.image)
            draw.rectangle([(xmin, ymin),
                        (xmax, ymax)],
                    outline=self.COLOR, width=3)
            font = ImageFont.load_default()
            text_labels='Press "y" to select all objects in MOSAIC.\n\n'
            text_labels+='Press "u" to unselect all objects in MOSAIC.\n\n'
            text_labels+='Press "t" for dropdown to change object name. \n\n'
            text_labels+='Press "h" to refresh.\n\n'
            text_labels+='Press "c" to create new name for object list below.\n\n'
            text_labels+='Press "d" to delete annotation for this object.\n\n'
            for label_j,int_i in self.label_dic.items():
                if int_i<10:
                    text_labels+='Press "{}" to change label to "{}"\n'.format(int_i,label_j)
                else:
                    text_labels+='Press "t" and select "{}" to change label to "{}"\n'.format(int_i,label_j)
            draw.text((0, 0),
                text_labels,
                fill='green', stroke_fill='blue',font=font)
            draw.text((xmin + 4, ymin + 4),
                '%s\n' % (label_i),
                fill=self.COLOR, font=font)
            self.img_i=cv2.cvtColor(np.array(self.image),cv2.COLOR_BGR2RGB)
            self.img_i_H=self.img_i.shape[1]
            self.img_i_W=self.img_i.shape[0]
            self.img_i_W_ratio=self.ROOT_W/self.img_i_W
            self.img_i_H_ratio=self.ROOT_H/self.img_i_H
            self.img_i_new_W=int(0.85*self.img_i_W_ratio*self.img_i_W)
            self.img_i_new_H=int(0.85*self.img_i_H_ratio*self.img_i_H)
            self.img_i=cv2.resize(self.img_i,(self.img_i_new_W,self.img_i_new_H))

            cv2.imshow('Selected Image.  Press "f" to fix.  Press "q" to quit.',self.img_i)
            plt.show()
    def onclick_show(self,event):
        #global df_i,axes_list,df,title_list,img_list
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
             ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        print('event.inaxes',event.inaxes)
        try:
            #cv2.destroyWindow('Selected Image.  Press "f" to fix.  Press "q" to quit.')
            cv2.destroyAllWindows()
        except:
            pass
        if event.button:
            self.ex=event.xdata
            self.ey=event.ydata
        if event.dblclick:
            self.ex=event.xdata
            self.ey=event.ydata
            self.df_dist=self.df.drop(['umap_input'],axis=1).copy()
            self.df_dist['distance']=self.df_dist['xmin'].copy()
            self.df_dist['distance']=0.0
            self.df_dist['distance']=self.df_dist['distance'].astype(np.float16)
            self.df_dist=self.df_dist[(abs(self.df_dist['emb_X']-self.ex)<0.25) & (abs(self.df_dist['emb_Y']-self.ey)<0.25)]
            self.df_dist['dx']=self.df_dist['emb_X'].copy()
            self.df_dist['dy']=self.df_dist['emb_Y'].copy()
            #for i,(x,y) in enumerate(zip(self.df['emb_X'],self.df['emb_Y'])):
            
            self.df_dist['dx']=self.ex-self.df_dist['emb_X']
            self.df_dist['dy']=self.ey-self.df_dist['emb_Y']
            self.df_dist['distance']=np.sqrt(self.df_dist['dx']**2+self.df_dist['dy']**2)
            self.df_dist=self.df_dist.sort_values(by='distance',ascending=True)
            #self.df.at[i,'distance']=np.sqrt(self.dx**2+self.dy**2)
            #self.df_to_fix_i=self.df.sort_values(by='distance',ascending=True).copy()
            self.index_bad=self.df_dist.iloc[0].name
            print(self.df_dist.head())
            self.image=Image.open(self.df_dist['path_jpeg_i'].iloc[0])
            xmin=self.df_dist['xmin'].iloc[0]
            xmax=self.df_dist['xmax'].iloc[0]
            ymin=self.df_dist['ymin'].iloc[0]
            ymax=self.df_dist['ymax'].iloc[0]
            label_i=self.df_dist['label_i'].iloc[0]
            draw=ImageDraw.Draw(self.image)
            draw.rectangle([(xmin, ymin),
                        (xmax, ymax)],
                    outline=self.COLOR, width=3)
            font = ImageFont.load_default()
            text_labels='Press "y" to select all objects in MOSAIC.\n\n'
            text_labels+='Press "u" to unselect all objects in MOSAIC.\n\n'
            text_labels+='Press "t" for dropdown to change object name. \n\n'
            text_labels+='Press "h" to refresh.\n\n'
            text_labels+='Press "c" to create new name for object list below.\n\n'
            text_labels+='Press "d" to delete annotation for this object.\n\n'
            for label_j,int_i in self.label_dic.items():
                if int_i<10:
                    text_labels+='Press "{}" to change label to "{}"\n'.format(int_i,label_j)
                else:
                    text_labels+='Press "t" and select "{}" to change label to "{}"\n'.format(int_i,label_j)

            draw.text((0, 0),
                text_labels,
                fill='green', stroke_fill='blue',font=font)
            draw.text((xmin + 4, ymin + 4),
                '%s\n' % (label_i),
                fill=self.COLOR, font=font)
            self.img_i=cv2.cvtColor(np.array(self.image),cv2.COLOR_BGR2RGB)
            self.img_i_H=self.img_i.shape[1]
            self.img_i_W=self.img_i.shape[0]
            self.img_i_W_ratio=self.ROOT_W/self.img_i_W
            self.img_i_H_ratio=self.ROOT_H/self.img_i_H
            self.img_i_new_W=int(0.85*self.img_i_W_ratio*self.img_i_W)
            self.img_i_new_H=int(0.85*self.img_i_H_ratio*self.img_i_H)
            self.img_i=cv2.resize(self.img_i,(self.img_i_new_W,self.img_i_new_H))

            cv2.imshow('Selected Image.  Press "f" to fix.  Press "q" to quit.',self.img_i)
            plt.show()

    def on_key_show_dxdy(self,event):
        print('you pressed', event.key, event.xdata, event.ydata)
        if event.key=='h':
            try:
                del self.a_xmin,self.a_xmax,self.a_ymin,self.a_ymax
                if _platform!='darwin':
                    plt.close() #edit sjs 5/28/2022
                else:
                    plt.close('all')
                self.inspect_mosaic=False
                self.run_selection=None
                self.close_window_mosaic=True
                cv2.destroyAllWindows()
                self.plot_dx_dy()   
            except:
                pass
        if event.key=='n' and self.inspect_mosaic==False:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        if event.key=='q' or event.key=='escape':
            self.close_window_mosaic=False
            self.plotting_dx_dy=False
            plt.close('all')
            try:
                cv2.destroyAllWindows()
            except:
                pass
            self.inspect_mosaic=False
            self.run_selection=None
        if event.key=='f':
            print('fixing it')
            self.df.at[self.index_bad,'checked']="bad"
            self.df.to_pickle(self.df_filename)
            self.df.to_csv(self.df_filename_csv,index=None)
            try:
                cv2.destroyAllWindows()
            except:
                pass
        if event.key=='m' and self.ex2 and self.ey2:
            if 'umap_input' in self.df.columns:
                self.df_sample=self.df.drop(['umap_input'],axis=1).copy()
            else:
                self.df_sample=self.df.copy()
            #print(self.df_dist.head())
            self.eymin=min(self.ey2,self.ey)
            self.eymax=max(self.ey2,self.ey)
            self.exmin=min(self.ex2,self.ex)
            self.exmax=max(self.ex2,self.ex)
            self.df_sample=self.df_sample[(self.df_sample['DX']>self.exmin) & (self.df_sample['DX']<self.exmax) & (self.df_sample['DY']>self.eymin) & (self.df_sample['DY']<self.eymax)]
            print(len(self.df_sample))
            self.inspect_mosaic=True
            self.close_window_mosaic=False
            print('looking at selection')
            self.k=0
            self.run_selection=self.look_at_selection()
            #self.inspect_mosaic=False
        if event.key=='d':
            try:
                print('deleting bounding box')
                xmin=str(self.df['xmin'][self.index_bad])
                xmax=str(self.df['xmax'][self.index_bad])
                ymin=str(self.df['ymin'][self.index_bad])
                ymax=str(self.df['ymax'][self.index_bad])
                print('xmin',xmin)
                print('xmax',xmax)
                print('ymin',ymin)
                print('ymax',ymax)
                label_k=self.df['label_i'][self.index_bad]
                path_anno_bad=self.df['path_anno_i'][self.index_bad]
                f=open(path_anno_bad,'r')
                f_read=f.readlines()
                f.close()
                f_new=[]
                start_line=0
                end_line=0
                self.df.drop(self.index_bad,axis=0)
                self.df=self.df.drop(self.index_bad,axis=0)
                # for ii,line in enumerate(f_read):
                #     if line.find(label_k)!=-1:
                #         print('found label_k',label_k)
                #         combo_i=f_read[ii-1:]
                #         combo_i="".join([w for w in combo_i])
                #         combo_i=combo_i.split('<object>')
                #         if combo_i[1].find(xmin)!=-1 and combo_i[1].find(xmax)!=-1 and combo_i[1].find(ymin)!=-1 and combo_i[1].find(ymax)!=-1:
                #             start_line=ii-1
                #             for jj,line_j in enumerate(f_read[ii:]):
                #                 if line_j.find('</object>')!=-1:
                #                     end_line=jj+ii
                #             f_new.append(line)
                #         else:
                #             f_new.append(line)
                #     else:
                #         f_new.append(line)

                for ii,line in enumerate(f_read):
                    if line.find(label_k)!=-1:   
                        combo_i=f_read[ii-1:]
                        combo_i="".join([w.replace('\n','') for w in combo_i])
                        combo_i=combo_i.split('<object>')
                        #print(len(combo_i),combo_i)
                        #if len(combo_i)>1:
                        if combo_i[1].find(xmin)!=-1 and combo_i[1].find(xmax)!=-1 and combo_i[1].find(ymin)!=-1 and combo_i[1].find(ymax)!=-1:
                            start_line=ii-1
                            for jj,line_j in enumerate(combo_i[1].split('</')):
                                if line_j.find('object>')!=-1:
                                    end_line=jj+ii
                                    print('found label_k',label_k)
                                    print('deleting bounding box')
                                    print('xmin',xmin)
                                    print('xmax',xmax)
                                    print('ymin',ymin)
                                    print('ymax',ymax)
                            f_new.append(line)
                        else:
                            f_new.append(line)
                        #else:
                        #    f_new.append(line)
                    else:
                        f_new.append(line)
                if end_line!=0:
                    f_new=f_new[:start_line]+f_new[end_line+1:]

                try:
                    f_new[0].find('annotation')!=-1
                except:
                    pprint(f_read)
                    assert f_new[0].find('annotation')!=-1
                # f_new=f_new[:start_line]+f_new[end_line+1:]
                # if f_new[0].find('annotation')==-1:
                #     f_old=f_new
                #     f_new=[]
                #     f_new.append('<annotation>\n')
                #     f_new=f_new+f_old
                #     print('fixed that before it forgot the first part')
                f=open(path_anno_bad,'w')
                [f.writelines(w) for w in f_new]
                f.close()  
                self.df.to_pickle(self.df_filename)
                self.df.to_csv(self.df_filename_csv,index=None)
                try:
                    cv2.destroyAllWindows()
                except:
                    pass 
            except:
                print('This ship has sailed, item not found.')
        if event.key=='c':
            self.popup()
            new_item_int=len(self.label_dic.keys())
            self.label_dic[self.w.value]=new_item_int
            self.rev_label_dic={v:k for k,v in self.label_dic.items()}
            self.draw_umap()

        for label_j,int_i in self.label_dic.items(): 
            print('label_j',label_j,'int_i',int_i)
            if event.key==str(int_i):
                print('fixing label')
                xmin=str(self.df['xmin'][self.index_bad])
                xmax=str(self.df['xmax'][self.index_bad])
                ymin=str(self.df['ymin'][self.index_bad])
                ymax=str(self.df['ymax'][self.index_bad])
                print('xmin',xmin)
                print('xmax',xmax)
                print('ymin',ymin)
                print('ymax',ymax)
                label_k=self.df['label_i'][self.index_bad]
                self.df.at[self.index_bad,'label_i']=self.rev_label_dic[int_i]
                self.df.at[self.index_bad,'label_i_int']=int_i
                f=open(self.df['path_anno_i'][self.index_bad],'r')
                f_read=f.readlines()
                f.close()
                f_new=[]
                for ii,line in enumerate(f_read):
                    if line.find(label_k)!=-1:
                        print('found label_k',label_k)
                        combo_i=f_read[ii-1:]
                        combo_i="".join([w for w in combo_i])
                        combo_i=combo_i.split('<object>')
                        #pprint(combo_i)
                        if combo_i[1].find(xmin)!=-1 and combo_i[1].find(xmax)!=-1 and combo_i[1].find(ymin)!=-1 and combo_i[1].find(ymax)!=-1:
                            print('fixing it')
                            print(line.replace(label_k,label_j))
                            f_new.append(line.replace(label_k,label_j))
                        else:
                            f_new.append(line)
                    else:
                        f_new.append(line)
                f=open(self.df['path_anno_i'][self.index_bad],'w')
                [f.writelines(w) for w in f_new]
                f.close()  
                self.df.to_pickle(self.df_filename)
                self.df.to_csv(self.df_filename_csv,index=None)
                try:
                    cv2.destroyAllWindows()
                except:
                    pass 
                break  

    def on_key_show(self,event):
        print('you pressed', event.key, event.xdata, event.ydata)
        if event.key=='h':
            try:
                del self.a_xmin,self.a_xmax,self.a_ymin,self.a_ymax
                if _platform!='darwin':
                    plt.close() #edit sjs 5/28/2022
                else:
                    plt.close('all')
                cv2.destroyAllWindows() #edit sjs 5/28/2022
                self.inspect_mosaic=False
                self.run_selection=None
                self.close_window_mosaic=True
                self.draw_umap()
            except:
                pass
            
                  
        if event.key=='n' and self.inspect_mosaic==False:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        if event.key=='q' or event.key=='escape':
            self.close_window_mosaic=False
            if _platform=='darwin':
                plt.close('all') #edit sjs 5/28/2022
                try:
                    cv2.destroyAllWindows() #edit sjs 5/28/2022
                except:
                    pass
            self.inspect_mosaic=False
            self.run_selection=None
            if _platform!='darwin': #edit sjs 5/28/2022
                plt.close('all')
        if event.key=='f':
            print('fixing it')
            self.df.at[self.index_bad,'checked']="bad"
            self.df.to_pickle(self.df_filename)
            self.df.to_csv(self.df_filename_csv,index=None)
            try:
                cv2.destroyAllWindows()
            except:
                pass
        if event.key=='m' and self.ex2 and self.ey2:
            self.df_sample=self.df.drop(['umap_input'],axis=1).copy()
            #print(self.df_dist.head())
            self.eymin=min(self.ey2,self.ey)
            self.eymax=max(self.ey2,self.ey)
            self.exmin=min(self.ex2,self.ex)
            self.exmax=max(self.ex2,self.ex)
            self.df_sample=self.df_sample[(self.df_sample['emb_X']>self.exmin) & (self.df_sample['emb_X']<self.exmax) & (self.df_sample['emb_Y']>self.eymin) & (self.df_sample['emb_Y']<self.eymax)]
            print(len(self.df_sample))
            self.inspect_mosaic=True
            self.close_window_mosaic=False
            print('looking at selection')
            self.k=0
            self.run_selection=self.look_at_selection()
            #self.inspect_mosaic=False
        if event.key=='c':
            self.popup()
            new_item_int=len(self.label_dic.keys())
            print('MAX value is: ',max(self.label_dic.values()))
            self.label_dic[self.w.value]=new_item_int
            self.rev_label_dic={v:k for k,v in self.label_dic.items()}
            self.draw_umap()



        if event.key=='d':
            try:
                print('deleting bounding box')
                xmin=str(self.df['xmin'][self.index_bad])
                xmax=str(self.df['xmax'][self.index_bad])
                ymin=str(self.df['ymin'][self.index_bad])
                ymax=str(self.df['ymax'][self.index_bad])
                print('xmin',xmin)
                print('xmax',xmax)
                print('ymin',ymin)
                print('ymax',ymax)
                label_k=self.df['label_i'][self.index_bad]
                path_anno_bad=self.df['path_anno_i'][self.index_bad]
                f=open(path_anno_bad,'r')
                f_read=f.readlines()
                f.close()
                f_new=[]
                start_line=0
                end_line=0
                self.df.drop(self.index_bad,axis=0)
                self.df=self.df.drop(self.index_bad,axis=0)
                # for ii,line in enumerate(f_read):
                #     if line.find(label_k)!=-1:
                #         print('found label_k',label_k)
                #         combo_i=f_read[ii-1:]
                #         combo_i="".join([w for w in combo_i])
                #         combo_i=combo_i.split('<object>')
                #         if combo_i[1].find(xmin)!=-1 and combo_i[1].find(xmax)!=-1 and combo_i[1].find(ymin)!=-1 and combo_i[1].find(ymax)!=-1:
                #             start_line=ii-1
                #             for jj,line_j in enumerate(f_read[ii:]):
                #                 if line_j.find('</object>')!=-1:
                #                     end_line=jj+ii
                #             f_new.append(line)
                #         else:
                #             f_new.append(line)
                #     else:
                #         f_new.append(line)
                # f_new=f_new[:start_line]+f_new[end_line+1:]

                for ii,line in enumerate(f_read):
                    if line.find(label_k)!=-1:   
                        combo_i=f_read[ii-1:]
                        combo_i="".join([w.replace('\n','') for w in combo_i])
                        combo_i=combo_i.split('<object>')
                        #print(len(combo_i),combo_i)
                        #if len(combo_i)>1:
                        if combo_i[1].find(xmin)!=-1 and combo_i[1].find(xmax)!=-1 and combo_i[1].find(ymin)!=-1 and combo_i[1].find(ymax)!=-1:
                            start_line=ii-1
                            for jj,line_j in enumerate(combo_i[1].split('</')):
                                if line_j.find('object>')!=-1:
                                    end_line=jj+ii
                                    print('found label_k',label_k)
                                    print('deleting bounding box')
                                    print('xmin',xmin)
                                    print('xmax',xmax)
                                    print('ymin',ymin)
                                    print('ymax',ymax)
                            f_new.append(line)
                        else:
                            f_new.append(line)
                        #else:
                        #    f_new.append(line)
                    else:
                        f_new.append(line)
                if end_line!=0:
                    f_new=f_new[:start_line]+f_new[end_line+1:]

                try:
                    f_new[0].find('annotation')!=-1
                except:
                    pprint(f_read)
                    assert f_new[0].find('annotation')!=-1
                f=open(path_anno_bad,'w')
                [f.writelines(w) for w in f_new]
                f.close()  
                self.df.to_pickle(self.df_filename)
                self.df.to_csv(self.df_filename_csv,index=None)
                try:
                    cv2.destroyAllWindows()
                except:
                    pass 
            except:
                print('This ship has sailed, item not found.')
        for label_j,int_i in self.label_dic.items(): 
            print('label_j',label_j,'int_i',int_i)
            if event.key==str(int_i):
                print('fixing label')
                xmin=str(self.df['xmin'][self.index_bad])
                xmax=str(self.df['xmax'][self.index_bad])
                ymin=str(self.df['ymin'][self.index_bad])
                ymax=str(self.df['ymax'][self.index_bad])
                print('xmin',xmin)
                print('xmax',xmax)
                print('ymin',ymin)
                print('ymax',ymax)
                label_k=self.df['label_i'][self.index_bad]
                self.df.at[self.index_bad,'label_i']=self.rev_label_dic[int_i]
                self.df.at[self.index_bad,'label_i_int']=int_i
                f=open(self.df['path_anno_i'][self.index_bad],'r')
                f_read=f.readlines()
                f.close()
                f_new=[]
                for ii,line in enumerate(f_read):
                    if line.find(label_k)!=-1:
                        print('found label_k',label_k)
                        combo_i=f_read[ii-1:]
                        combo_i="".join([w for w in combo_i])
                        combo_i=combo_i.split('<object>')
                        #pprint(combo_i)
                        if combo_i[1].find(xmin)!=-1 and combo_i[1].find(xmax)!=-1 and combo_i[1].find(ymin)!=-1 and combo_i[1].find(ymax)!=-1:
                            print('fixing it')
                            print(line.replace(label_k,label_j))
                            f_new.append(line.replace(label_k,label_j))
                        else:
                            f_new.append(line)
                    else:
                        f_new.append(line)
                f=open(self.df['path_anno_i'][self.index_bad],'w')
                [f.writelines(w) for w in f_new]
                f.close()  
                self.df.to_pickle(self.df_filename)
                self.df.to_csv(self.df_filename_csv,index=None)
                try:
                    cv2.destroyAllWindows()
                except:
                    pass 
                break    
            
    def look_at_selection(self):

        # if 'emb_X' in self.df.columns:
        #     self.df=self.df.sort_values(['emb_X','emb_Y','label_dist'],ascending=[False,False,False]).reset_index().drop('index',axis=1)
        self.df_i=self.df_sample.copy()
        self.selection_list={}
        DX=int(np.ceil(np.sqrt(len(self.df_i))))
        DY=int(np.ceil(np.sqrt(len(self.df_i))))
        print(len(self.df_i))
        
        for _ in tqdm(range(1)):
            print('self.close_window_mosaic==',self.close_window_mosaic)
            if self.close_window_mosaic==True:
                self.close_window_mosaic=False
                break
            if self.go_to_next==False and self.k!=0:
                break
            self.go_to_next=False
            self.axes_list=[]
            self.title_list=[]
            self.img_list=[]
            self.dic={}

            self.start=0#self.end
            self.end=self.start+len(self.df_i)#self.MOSAIC_NUM
            if self.end>len(self.df_i):
                self.end=len(self.df_i)
            self.fig_i=plt.figure(figsize=(self.FIGSIZE_W,self.FIGSIZE_H),num='Showing {}/{} chips.  Press "q" to quit.  Press "n" for next.'.format(self.end,len(self.df_i)))
            self.fig_i.set_size_inches((self.FIGSIZE_INCH_W, self.FIGSIZE_INCH_W))
            self.fig_i_cid = self.fig_i.canvas.mpl_connect('button_press_event', self.onclick_select_mosaic)
            self.fig_i_cidk = self.fig_i.canvas.mpl_connect('key_press_event', self.on_key_mosaic)

            j=0
            for i in range(self.start,self.end):
                self.dic[j]=i
                j+=1
                self.jpg_i=plt.imread(self.df_i['path_jpeg_i'].iloc[i])
                self.xmin_i=self.df_i['xmin'].iloc[i]
                self.xmax_i=self.df_i['xmax'].iloc[i]
                self.ymin_i=self.df_i['ymin'].iloc[i]
                self.ymax_i=self.df_i['ymax'].iloc[i]
                self.longest=max(self.ymax_i-self.ymin_i,self.xmax_i-self.xmin_i)

                self.chip_i=self.jpg_i[self.ymin_i:self.ymax_i,self.xmin_i:self.xmax_i,:]
                try:
                    self.chip_square_i=Image.fromarray(self.chip_i) 
                except:
                    print(self.df['path_jpeg_i'].iloc[i])
                    self.chip_square_i=Image.fromarray(self.chip_i)   

                self.chip_square_i=self.chip_square_i.resize((self.W,self.H),Image.ANTIALIAS)
                self.chip_square_i=np.array(self.chip_square_i)
                if j==1:
                    self.A_ind=self.df_i.iloc[i].name
                    self.B_ind=self.df_i.iloc[i].name
                    self.grayA=cv2.cvtColor(self.chip_square_i,cv2.COLOR_BGR2GRAY) 
                    self.grayB=cv2.cvtColor(self.chip_square_i,cv2.COLOR_BGR2GRAY) 
                else:
                    self.A_ind=self.df_i.iloc[i-1].name
                    self.B_ind=self.df_i.iloc[i].name  
                    self.grayA=self.grayB
                    self.grayB=cv2.cvtColor(self.chip_square_i,cv2.COLOR_BGR2GRAY) 
                  
                self.axes_list.append(self.fig_i.add_subplot(DX,DY,i+1-self.start))
                plt.subplots_adjust(wspace=0.2,hspace=0.5)
                
                self.img_list.append(self.chip_square_i)
                plt.imshow(self.chip_square_i)
                plt.axis('off')

                self.title_list.append(plt.title(self.df_i.iloc[i].label_i,fontsize='5',color='blue'))
                if i==len(self.df_i):
                    break
            plt.show()
            self.k+=1
            #self.df.to_pickle(self.df_filename) #TBD
    def look_at_objects(self):
        
        self.go_to_next=True
        self.close_window=False
        if 'umap_input' in self.df.columns:
            self.df_i=self.df.drop('umap_input',axis=1)
        else:
            self.df_i=self.df

        print(len(self.df_i))
        
        for k,jpg_i in tqdm(enumerate(self.df_i['path_jpeg_i'].unique())):
            self.df_j=self.df_i[self.df_i['path_jpeg_i']==jpg_i].copy()

            if self.close_window==True:
                break
            self.go_to_next=False
            self.axes_list=[]
            self.title_list=[]
            self.img_list=[]
            self.dic={}
            self.index_i_fix=[]

            self.fig_i=plt.figure(figsize=(self.FIGSIZE_W,self.FIGSIZE_H),num='Showing {}.  Press "f" fix, "u" unfix, "q" quit, "n" for next.'.format(os.path.basename(jpg_i)))
            self.fig_i.set_size_inches((self.FIGSIZE_INCH_W, self.FIGSIZE_INCH_W))
            self.cidk = self.fig_i.canvas.mpl_connect('key_press_event', self.on_key_object)
            self.image=Image.open(jpg_i)
            
            for index_i in list(self.df_j[self.df_j['path_jpeg_i']==jpg_i].index):
                xmin=self.df_j['xmin'].loc[index_i]
                xmax=self.df_j['xmax'].loc[index_i]
                ymin=self.df_j['ymin'].loc[index_i]
                ymax=self.df_j['ymax'].loc[index_i]
                label_i=self.df_j['label_i'].loc[index_i]
                draw=ImageDraw.Draw(self.image)
                draw.rectangle([(xmin, ymin),
                            (xmax, ymax)],
                        outline=self.COLOR, width=3)
                font = ImageFont.load_default()
                draw.text((xmin + 4, ymin + 4),
                    '%s\n' % (label_i),
                    fill=self.COLOR, font=font)
                self.img_i=cv2.cvtColor(np.array(self.image),cv2.COLOR_BGR2RGB)
                self.img_i_H=self.img_i.shape[1]
                self.img_i_W=self.img_i.shape[0]
                self.img_i_W_ratio=self.ROOT_W/self.img_i_W
                self.img_i_H_ratio=self.ROOT_H/self.img_i_H
                self.img_i_new_W=int(0.95*self.img_i_W_ratio*self.img_i_W)
                self.img_i_new_H=int(0.95*self.img_i_H_ratio*self.img_i_H)
                self.img_i=cv2.resize(self.img_i,(self.img_i_new_W,self.img_i_new_H))
                self.index_i_fix.append(index_i)
            #self.title_list.append(plt.title(os.path.basename(jpg_i),fontsize='5',color='blue'))
           
            plt.imshow(cv2.cvtColor(self.img_i, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()
            #cv2.imshow('Selected Image.  Press "q" to close window',self.img_i)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #plt.show()

            #self.df.to_pickle(self.df_filename)
            #self.df.to_csv(self.df_filename_csv,index=None)

    def look_at_target(self,target_i):
        self.target_i=target_i
        # if 'emb_X' in self.df.columns:
        #     self.df=self.df.sort_values(['emb_X','emb_Y','label_dist'],ascending=[False,False,False]).reset_index().drop('index',axis=1)
        if 'umap_input' in self.df.columns:
            self.df_i=self.df[(self.df['label_i']==self.target_i) & (self.df['checked']=='')].drop('umap_input',axis=1)
        else:
            self.df_i=self.df[(self.df['label_i']==self.target_i) & (self.df['checked']=='')]

        print(len(self.df_i))
        
        for k in tqdm(range(1+int(np.ceil(len(self.df_i)//self.MOSAIC_NUM)))):

            if self.close_window==True:
                break
            self.go_to_next=False
            self.axes_list=[]
            self.title_list=[]
            self.img_list=[]
            self.dic={}
            if k==0:
                self.start=0
                self.end=self.MOSAIC_NUM
                if self.end>len(self.df_i):
                    self.end=len(self.df_i)
            else:
                self.start=self.end
                self.end=self.start+self.MOSAIC_NUM
                if self.end>len(self.df_i):
                    self.end=len(self.df_i)
            self.fig_i=plt.figure(figsize=(self.FIGSIZE_W,self.FIGSIZE_H),num='Showing {}/{} "{}" chips.  Press "q" to quit.  Press "n" for next.'.format(self.end,len(self.df_i),self.target_i,))
            self.fig_i.set_size_inches((self.FIGSIZE_INCH_W, self.FIGSIZE_INCH_W))
            self.cid = self.fig_i.canvas.mpl_connect('button_press_event', self.onclick)
            self.cidk = self.fig_i.canvas.mpl_connect('key_press_event', self.on_key)
            j=0
            for i in range(self.start,self.end):
                self.dic[j]=i
                j+=1
                self.jpg_i=plt.imread(self.df_i['path_jpeg_i'].iloc[i])
                self.xmin_i=self.df_i['xmin'].iloc[i]
                self.xmax_i=self.df_i['xmax'].iloc[i]
                self.ymin_i=self.df_i['ymin'].iloc[i]
                self.ymax_i=self.df_i['ymax'].iloc[i]
                self.longest=max(self.ymax_i-self.ymin_i,self.xmax_i-self.xmin_i)
              
                # try:
                #     self.chip_i=self.jpg_i[self.ymin_i-5:self.ymin_i+self.longest+5,self.xmin_i-5:self.xmin_i+self.longest+5,:]
                #     self.chip_square_i=Image.fromarray(self.chip_i)
                # except:
                #     try:
                #         self.chip_i=self.jpg_i[self.ymin_i:self.ymin_i+self.longest,self.xmin_i:self.xmin_i+self.longest,:]
                #         self.chip_square_i=Image.fromarray(self.chip_i)
                #     except:
                #         self.chip_i=self.jpg_i[self.ymin_i:self.ymax_i,self.xmin_i:self.xmax_i,:]
                #         self.chip_square_i=Image.fromarray(self.chip_i)     

                self.chip_i=self.jpg_i[self.ymin_i:self.ymax_i,self.xmin_i:self.xmax_i,:]
                self.chip_square_i=Image.fromarray(self.chip_i) 


                self.chip_square_i=self.chip_square_i.resize((self.W,self.H),Image.ANTIALIAS)
                self.chip_square_i=np.array(self.chip_square_i)
                self.df.at[self.df_i.iloc[i].name,'checked']='good'
                if j==1:
                    self.A_ind=self.df_i.iloc[i].name
                    self.B_ind=self.df_i.iloc[i].name
                    self.grayA=cv2.cvtColor(self.chip_square_i,cv2.COLOR_BGR2GRAY) 
                    self.grayB=cv2.cvtColor(self.chip_square_i,cv2.COLOR_BGR2GRAY) 
                else:
                    self.A_ind=self.df_i.iloc[i-1].name
                    self.B_ind=self.df_i.iloc[i].name  
                    self.grayA=self.grayB
                    self.grayB=cv2.cvtColor(self.chip_square_i,cv2.COLOR_BGR2GRAY) 
                if self.useSSIM:
                    (self.score,self.diff)=ssim(self.grayA,self.grayB,full=True)
                    self.diff=(self.diff*255).astype("uint8")
                  
                self.axes_list.append(self.fig_i.add_subplot(self.DX,self.DY,i+1-self.start))
                plt.subplots_adjust(wspace=0.2,hspace=0.5)
                
                self.img_list.append(self.chip_square_i)
                plt.imshow(self.chip_square_i)
                plt.axis('off')
                if self.useSSIM:
                    if self.score<0.20:
                        self.title_list.append(plt.title("i={}, score={}%".format(i,int(100*np.round(self.score,2))),fontsize='5',color='red'))
                    else:
                        self.title_list.append(plt.title("",fontsize='5',color='red'))
                else:
                    self.title_list.append(plt.title("",fontsize='5',color='red'))
                if i==len(self.df_i):
                    break
            plt.show()
            #while self.go_to_next==False and self.close_window==False:
            #    time.sleep(0.1)
            #    pass
            self.df.to_pickle(self.df_filename)
            self.df.to_csv(self.df_filename_csv,index=None)
    def get_umap_input(self):
        self.df=self.df.reset_index().drop('index',axis=1)
        for i in tqdm(range(len(self.df))):
            self.jpg_i=plt.imread(self.df['path_jpeg_i'].iloc[i])
            self.xmin_i=self.df['xmin'].iloc[i]
            self.xmax_i=self.df['xmax'].iloc[i]
            self.ymin_i=self.df['ymin'].iloc[i]
            self.ymax_i=self.df['ymax'].iloc[i]
            self.longest=max(self.ymax_i-self.ymin_i,self.xmax_i-self.xmin_i)
            # try:
            #     self.chip_i=self.jpg_i[self.ymin_i-5:self.ymin_i+self.longest+5,self.xmin_i-5:self.xmin_i+self.longest+5,:]
            #     self.chip_square_i=Image.fromarray(self.chip_i)
            # except:
            #     try:
            #         self.chip_i=self.jpg_i[self.ymin_i:self.ymin_i+self.longest,self.xmin_i:self.xmin_i+self.longest,:]
            #         self.chip_square_i=Image.fromarray(self.chip_i)
            #     except:
            #         self.chip_i=self.jpg_i[self.ymin_i:self.ymax_i,self.xmin_i:self.xmax_i,:]
            #         self.chip_square_i=Image.fromarray(self.chip_i)    
            def to_rgb2(im):
                w,h=im.shape
                ret=np.empty((w,h,3),dtype=np.uint8)
                ret[:,:,:]=im[:,:,np.newaxis]
                return ret
            if len(self.jpg_i.shape)!=3:
                self.jpg_i=to_rgb2(self.jpg_i)
            if self.ymin_i==self.ymax_i:
                self.ymax_i+=1
            if self.xmin_i==self.xmax_i:
                self.xmax_i+=1
            self.chip_i=self.jpg_i[self.ymin_i:self.ymax_i,self.xmin_i:self.xmax_i,:]
            try:
                self.chip_square_i=Image.fromarray(self.chip_i) 
            except:
                print(self.df['path_jpeg_i'].iloc[i])
                self.chip_square_i=Image.fromarray(self.chip_i)           
            self.chip_square_i=self.chip_square_i.resize((self.W,self.H),Image.ANTIALIAS)
            self.chip_square_i=np.array(self.chip_square_i)
            self.df.at[i,'umap_input']=self.chip_square_i
        self.df.to_pickle(self.df_filename)
        self.df.to_csv(self.df_filename_csv,index=None)
    def get_umap_input_quick(self,i):
        self.jpg_i=plt.imread(self.df['path_jpeg_i'].iloc[i])
        self.xmin_i=self.df['xmin'].iloc[i]
        self.xmax_i=self.df['xmax'].iloc[i]
        self.ymin_i=self.df['ymin'].iloc[i]
        self.ymax_i=self.df['ymax'].iloc[i]
        self.longest=max(self.ymax_i-self.ymin_i,self.xmax_i-self.xmin_i)

        def to_rgb2(im):
            w,h=im.shape
            ret=np.empty((w,h,3),dtype=np.uint8)
            ret[:,:,:]=im[:,:,np.newaxis]
            return ret
        if len(self.jpg_i.shape)!=3:
            self.jpg_i=to_rgb2(self.jpg_i)
        if self.ymin_i==self.ymax_i:
            self.ymax_i+=1
        if self.xmin_i==self.xmax_i:
            self.xmax_i+=1
        self.chip_i=self.jpg_i[self.ymin_i:self.ymax_i,self.xmin_i:self.xmax_i,:]
        try:
            self.chip_square_i=Image.fromarray(self.chip_i) 
        except:
            print(self.df['path_jpeg_i'].iloc[i])
            self.chip_square_i=Image.fromarray(self.chip_i)           
        self.chip_square_i=self.chip_square_i.resize((self.W,self.H),Image.ANTIALIAS)
        self.chip_square_i=np.array(self.chip_square_i)
        #self.df.at[i,'umap_input']=''
        return self.chip_square_i
    def get_umap_input_quick_range(self,istart,iend,fl_pkl):
        import pickle
        f=open(fl_pkl,'wb')
        flattened=[]
        for i in range(istart,iend):
            jpg_i=plt.imread(self.df['path_jpeg_i'].iloc[i])
            xmin_i=self.df['xmin'].iloc[i]
            xmax_i=self.df['xmax'].iloc[i]
            ymin_i=self.df['ymin'].iloc[i]
            ymax_i=self.df['ymax'].iloc[i]

            longest=max(ymax_i-ymin_i,xmax_i-xmin_i)

            def to_rgb2(im):
                w,h=im.shape
                ret=np.empty((w,h,3),dtype=np.uint8)
                ret[:,:,:]=im[:,:,np.newaxis]
                return ret
            if len(jpg_i.shape)!=3:
                jpg_i=to_rgb2(jpg_i)
            if ymin_i==ymax_i:
                ymax_i+=1
            if xmin_i==xmax_i:
                xmax_i+=1
            chip_i=jpg_i[ymin_i:ymax_i,xmin_i:xmax_i,:]
            try:
                chip_square_i=Image.fromarray(chip_i) 
            except:
                print(self.df['path_jpeg_i'].iloc[i])
 
                print('jpg_i.shape',jpg_i.shape)
                print('chip_i.shape',chip_i.shape)
                print('xmin_i',xmin_i)
                print('xmax_i',xmax_i)
                print('ymin_i',ymin_i)
                print('ymax_i',ymax_i)
                chip_i=chip_i*255
                chip_i=chip_i.astype(np.uint8)
                chip_square_i=Image.fromarray(chip_i)
                #plt.imshow(chip_i)
                #plt.show()

                #chip_square_i=Image.fromarray(chip_i[:,:,0])

                            
            chip_square_i=chip_square_i.resize((self.W,self.H),Image.ANTIALIAS)
            chip_square_i=np.array(chip_square_i)
            #self.df.at[i,'umap_input']=''
            flattened.append(np.array(chip_square_i).flatten())
   
        pickle.dump(flattened,f)
        f.close()
    def breakup_df(self):
        print(len(self.df))
        self.SPLIT_NUM=1000
        prefix_uniques=[os.path.basename(w).split('.')[0] for w in self.df['path_anno_i'].unique()]
        if len(self.df)<self.SPLIT_NUM:
            self.SPLIT_NUM=len(self.df)
        count=0
        for i,(xml_i,jpeg_i) in tqdm(enumerate(zip(self.df['path_anno_i'],self.df['path_jpeg_i']))):
            if count%self.SPLIT_NUM==0:
                xml_list_i=[]
                jpeg_list_i=[]
                start_i=count
                end_i=start_i+self.SPLIT_NUM-1
                prefix_i=os.path.basename(xml_i) #.split('/')[-1]
                prefix_j=xml_i.replace(prefix_i,'').rstrip('/').split('/')[-1]
                base_path_i=xml_i.replace(prefix_i,'').replace(prefix_j,'').rstrip('/')+'__SPLIT_NUM_{}'.format(self.SPLIT_NUM)
                if os.path.exists(base_path_i)==False:
                    os.makedirs(base_path_i)
                new_path_i=os.path.join(base_path_i,"{}_{}".format(start_i,end_i))
                if os.path.exists(new_path_i)==False:
                    os.makedirs(new_path_i)
                new_path_i_anno=os.path.join(new_path_i,'Annotations')
                new_path_i_jpeg=os.path.join(new_path_i,'JPEGImages')
                if os.path.exists(new_path_i_anno)==False:
                    os.makedirs(new_path_i_anno)
                if os.path.exists(new_path_i_jpeg)==False:
                    os.makedirs(new_path_i_jpeg)
            xml_list_i.append(xml_i)
            jpeg_list_i.append(jpeg_i)
            shutil.copy(xml_i,new_path_i_anno)
            shutil.copy(jpeg_i,new_path_i_jpeg)
            if (count+1)%self.SPLIT_NUM==0 or (count+1)==len(self.df):
                if (count+1)==self.SPLIT_NUM:
                    self.path_JPEGImages=new_path_i_jpeg
                    self.path_Annotations=new_path_i_anno
                df_i=self.df[self.df.path_anno_i.isin(xml_list_i)].copy()
                df_i=df_i.reset_index().drop('index',axis=1)
                df_i_filename=os.path.join(new_path_i,'df_{}.pkl'.format(os.path.basename(new_path_i)))
                df_i.to_pickle(df_i_filename)
            count+=1
    def draw_umap(self):
        if _platform!='darwin':
            plt.clf() #edit sjs 5/28/2022
        i=0
        for label in self.df['label_i'].unique():
            if label not in self.label_dic.keys():
                self.label_dic[label]=i
                i+=1       
        self.rev_label_dic={v:k for k,v in self.label_dic.items()}
        self.df.to_pickle(self.df_filename)
        self.df.to_csv(self.df_filename_csv,index=None)
        self.fig_j=plt.figure(figsize=(self.FIGSIZE_W,self.FIGSIZE_H),num='UMAP.  "Double Click" to inspect.  Press "q" to quit.    Press "m" for MOSAIC.')
        self.fig_j.set_size_inches((self.FIGSIZE_INCH_W, self.FIGSIZE_INCH_W))
        self.cidj = self.fig_j.canvas.mpl_connect('button_press_event', self.onclick_show)
        self.cidkj = self.fig_j.canvas.mpl_connect('key_press_event', self.on_key_show)
        self.cidkjm = self.fig_j.canvas.mpl_connect('button_release_event',self.release_show)
        self.close_window_mosaic=False

        plt.rcParams['axes.facecolor'] = 'gray'
        plt.grid(c='white')
        text_labels='Press "y" to select all objects in MOSAIC.\n\n'
        text_labels+='Press "u" to unselect all objects in MOSAIC.\n\n'
        text_labels+='Press "t" for dropdown to change object name. \n\n'
        text_labels+='Press "h" to refresh.\n\n'
        text_labels+='Press "c" to create new name for object list below.\n\n'
        text_labels+='Press "d" to delete annotation for this object.\n\n'

        for label_j,int_i in self.label_dic.items():
            if int_i<10:
                text_labels+='Press "{}" to change label to "{}"\n'.format(int_i,label_j)
            else:
                text_labels+='Press "t" and select "{}" to change label to "{}"\n'.format(int_i,label_j)
        plt.xlabel(text_labels)
        plt.scatter(self.df['emb_X'],self.df['emb_Y'],c=self.df['label_i_int'],cmap='Spectral',s=5)
        self.ex=min(self.df['emb_X'])
        self.ex2=max(self.df['emb_X'])
        self.ey=min(self.df['emb_Y'])
        self.ey2=max(self.df['emb_Y'])
        self.gca=plt.gca()
        self.gca.set_aspect('equal','datalim')
        try:
            plt.ylabel('Press "h" to zoom out')
            plt.xlim([self.a_xmin,self.a_xmax])
            plt.ylim([self.a_ymin,self.a_ymax])
        except:
            pass
        if len(self.df['label_i_int'].unique())>1:
            plt.colorbar(boundaries=np.arange(len(self.df['label_i_int'].unique())+1)-0.5).set_ticks(np.arange(len(self.df['label_i_int'].unique())))
        plt.tight_layout()
        plt.show()
    def get_umap_output(self):
        self.df=self.df.dropna(axis=1)
        import pickle
        if 'emb_X' not in self.df.columns:
            if 'umap_input' not in self.df.columns:
                self.df['umap_input']=self.df['path_anno_i'].copy()
                self.df['umap_input']=''  
                #self.get_umap_input()
            #self.reducer=umap.UMAP(random_state=42)
            #self.flattened=[np.array(w).flatten() for w in self.df.umap_input]
            #self.flattened=[np.array(self.get_umap_input_quick(w)).flatten() for w in self.df.index]
            print('CONVERTING CHIPS to flattened UMAP array')
            self.flattened=[]
            #count_w=0
            #self.fl_pkl='flt_{}.pkl'.format(count_w)
            self.fl_pkls=[]
            #self.fl_pkls.append(self.fl_pkl)
            #f=open(self.fl_pkl,'wb')
            end_of_index=list(self.df.index)[-1]+1
            from multiprocessing import Process
            #import multiprocessing
            #pool=multiprocessing.Pool(5)
            processes=[]
            CHUNK=200
            if multiprocessing.cpu_count()>1:
                NUM_PROCESS=multiprocessing.cpu_count()-1
            else:
                NUM_PROCESS=1

            print('There are a total of {} items to go through at CHUNK={}'.format(end_of_index,CHUNK))
            if os.path.exists('tmp')==False:
                os.makedirs('tmp')
            for j,i in tqdm(enumerate(range(0,end_of_index,CHUNK))):
                fl_pkl='tmp/flt_{}_{}.pkl'.format(i,min(end_of_index,j*CHUNK+CHUNK))
                self.fl_pkls.append(fl_pkl)
                p=Process(target=self.get_umap_input_quick_range,args=(i,min(end_of_index,j*CHUNK+CHUNK),fl_pkl))
                processes.append(p)
                #processes.append([i,min(end_of_index,j*500+500),fl_pkl])
                p.start()
                if (j%NUM_PROCESS==0 and j!=0) or min(end_of_index,j*CHUNK+CHUNK)==end_of_index:
                    print('\nStarting {} new proccess for {} of {}\n'.format(NUM_PROCESS,j,end_of_index//CHUNK))
                    for process_i in tqdm(processes):
                        process_i.join()
                    processes=[]
            #proccess=tuple(processes)
            #pool.map(self.get_umap_input_quick_range,processes)
            #pool.close()
            #pool.join()
            # for process_i in tqdm(processes):
            #     process_i.join()
            # for w in tqdm(self.df.index):
            #     self.flattened.append(np.array(self.get_umap_input_quick(w)).astype(np.uint8).flatten())
            #     if w%500==0 and w!=0:
            #         pickle.dump(self.flattened,f)
            #         self.flattened=[]
            #         f.close()
            #         count_w=w
            #         if w!=end_of_index:
            #             self.fl_pkl='flt_{}.pkl'.format(count_w)
            #             self.fl_pkls.append(self.fl_pkl)
            #             f=open(self.fl_pkl,'wb')
            # try:
            #     pickle.dump(self.flattened,f)
            #     f.close()
            # except:
            #     pass
            #self.reducer=umap.UMAP(random_state=42)
            MAX_FLAT_LEN=10000
            RESET=True
            self.reducer=umap.UMAP(random_state=42,low_memory=True)
            for i,file_i in tqdm(enumerate(self.fl_pkls)):

                try:
                    f=open(file_i,'rb')
                    if RESET:
                        #self.reducer=umap.UMAP(random_state=42)
                        self.flattened=np.array(pickle.load(f))
                        RESET=False
                    else:                      
                        self.flattened=np.concatenate((self.flattened,np.array(pickle.load(f))),axis=0)
                    if self.flattened.shape[0]%MAX_FLAT_LEN==0 or i==len(self.fl_pkls)-1:
                        print('FITTING to the flattened UMAP array')
                        self.reducer.fit(self.flattened.astype(np.uint8))
                        print('TRANSFORMING to the flattened UMAP array')
                        try:
                            self.embedding=np.concatenate((self.embedding,np.array(self.reducer.transform(self.flattened.astype(np.uint8)))),axis=0)
                        except:
                            self.embedding=np.array(self.reducer.transform(self.flattened.astype(np.uint8)))
                        RESET=True
                    f.close()
                except:
                    pass
                os.remove(file_i)

            #self.embedding=np.array(self.embedding)
            #self.embedding=np.expand_dims(self.embedding,axis=-1)
            # print('FITTING to the flattened UMAP array')
            # self.reducer.fit(self.flattened.astype(np.uint8))
            # print('TRANSFORMING to the flattened UMAP array')
            # self.embedding=self.reducer.transform(self.flattened.astype(np.uint8))
            print('self.embedding.shape',self.embedding.shape)


            self.df.to_pickle(self.df_filename)
            self.df.to_csv(self.df_filename_csv,index=None)
            # self.reducer=umap.UMAP(random_state=42)
            # print('FITTING to the flattened UMAP array')
            # self.reducer.fit(self.flattened)
            # print('TRANSFORMING to the flattened UMAP array')
            # self.embedding=self.reducer.transform(self.flattened)
            self.df['emb_X']=self.df['label_i'].copy()
            self.df['emb_Y']=self.df['label_i'].copy()
            self.df['emb_X']=[w for w in self.embedding[:,0]]
            self.df['emb_Y']=[w for w in self.embedding[:,1]]
            self.df['emb_X']=self.df['emb_X'].astype(np.float16)
            self.df['emb_Y']=self.df['emb_Y'].astype(np.float16)
            i=0
            for label in self.df['label_i'].unique():
                if label not in self.label_dic.keys():
                    self.label_dic[label]=i
                    i+=1

            self.df['label_i_int']=self.df['label_i'].copy()
            self.df['label_i_int']=[self.label_dic[w] for w in self.df['label_i']]
            self.df['label_dist']=self.df['emb_X'].copy()
            unique_labels=list(self.df.label_i.unique())
            unique_labels_dic={}
            unique_dfs={}
            print('Operating on all unique labels for embeddings')
            for unique_label in tqdm(unique_labels):
                print('unique_label',unique_label)
                df_i=self.df[self.df.label_i==unique_label].copy()
                df_i=df_i.reset_index().drop('index',axis=1)
                df_i['avg_X_i']=df_i['emb_X'].mean()
                df_i['avg_Y_i']=df_i['emb_Y'].mean()
                df_i['label_dist']=df_i[['emb_X','avg_X_i','emb_Y','avg_Y_i']].apply(lambda x: np.sqrt((x[0]-x[1])**2+(x[2]-x[3])**2),axis=1)
                unique_dfs[unique_label]=df_i
                #print(unique_dfs[unique_label].head())
            print('Appending each unique label pandas dataframe')
            for i,unique_df in tqdm(enumerate(unique_dfs.values())):
                if i==0:
                    self.df=unique_df
                else:
                    self.df=self.df.append(unique_df,ignore_index=True)
            self.df=self.df.reset_index().drop('index',axis=1)
            del unique_dfs
            del unique_labels_dic
            del unique_labels
            # for row in tqdm(range(len(self.df))):
            #     label_i=self.df.iloc[row].label_i
            #     avg_X_i=self.df[self.df.label_i==label_i]['emb_X'].mean()
            #     avg_Y_i=self.df[self.df.label_i==label_i]['emb_Y'].mean()
            #     X_i=self.df.iloc[row].emb_X
            #     Y_i=self.df.iloc[row].emb_Y
            #     d_X_i=X_i-avg_X_i
            #     d_Y_i=Y_i-avg_Y_i
            #     dist_i=np.sqrt(d_X_i**2+d_Y_i**2)
            #     self.df.at[row,'label_dist']=dist_i
        self.draw_umap()



class App:
    def __init__(self,root_tk,path_Annotations='None',path_JPEGImages='None'):
        self.root=root_tk
        self.root.bind('<Escape>',self.close)
        self.root.wm_iconphoto(False,ImageTk.PhotoImage(Image.open("resources/icons/appM.png")))
        self.icon_breakup=ImageTk.PhotoImage(Image.open('resources/icons/breakup.png'))
        self.icon_folder=ImageTk.PhotoImage(Image.open("resources/icons/file.png"))
        self.icon_load=ImageTk.PhotoImage(Image.open('resources/icons/load.png'))
        self.icon_create=ImageTk.PhotoImage(Image.open('resources/icons/create.png'))
        self.icon_analyze=ImageTk.PhotoImage(Image.open('resources/icons/analyze.png'))
        self.icon_move=ImageTk.PhotoImage(Image.open('resources/icons/move.png'))
        self.icon_labelImg=ImageTk.PhotoImage(Image.open('resources/icons/labelImg.png'))
        self.icon_map=ImageTk.PhotoImage(Image.open('resources/icons/map.png'))
        self.icon_merge=ImageTk.PhotoImage(Image.open('resources/icons/merge.png'))
        self.icon_clear_fix=ImageTk.PhotoImage(Image.open('resources/icons/clear.png'))
        self.icon_clear_checked=ImageTk.PhotoImage(Image.open('resources/icons/clear_checked.png'))
        self.icon_save_settings=ImageTk.PhotoImage(Image.open('resources/icons/save.png'))
        self.icon_scatter=ImageTk.PhotoImage(Image.open('resources/icons/scatter.png'))
        self.icon_filter=ImageTk.PhotoImage(Image.open('resources/icons/filter.png'))
        self.icon_video=ImageTk.PhotoImage(Image.open('resources/icons/test_mp4.png'))
        if path_JPEGImages=='None':
            self.path_JPEGImages=DEFAULT_SETTINGS.path_JPEGImages #r'/Volumes/One Touch/Images_gdrive/Drone_Images/Training/JPEGImages'
        else:
            self.path_JPEGImages=path_JPEGImages
        if path_Annotations=='None':
            self.path_Annotations=DEFAULT_SETTINGS.path_Annotations #r'/Volumes/One Touch/Images_gdrive/Drone_Images/Training/Annotations'
        else:
            self.path_Annotations=path_Annotations
        self.path_labelImg=DEFAULT_SETTINGS.path_labelImg #r'/Volumes/One Touch/labelImg-Custom/labelImg.py'
        self.jpeg_selected=False #after  user has opened from folder dialog the jpeg folder, this returns True
        self.anno_selected=False #after user has opened from folder dialog the annotation folder, this returns True
        self.create_move_Anno_JPEG()
        self.root_H=int(self.root.winfo_screenheight()*0.95)
        self.root_W=int(self.root.winfo_screenwidth()*0.95)
        self.root.geometry(str(self.root_W)+'x'+str(self.root_H))
        self.root.title("MOSAIC Chip Sorter")
        self.root_bg=DEFAULT_SETTINGS.root_bg#'black'
        self.root_fg=DEFAULT_SETTINGS.root_fg#'lime'
        self.predefined_classes=DEFAULT_SETTINGS.predefined_classes
        self.PYTHON_PATH=DEFAULT_SETTINGS.PYTHON_PATH
        self.canvas_columnspan=DEFAULT_SETTINGS.canvas_columnspan
        self.canvas_rowspan=DEFAULT_SETTINGS.canvas_rowspan
        self.MOSAIC_NUM=DEFAULT_SETTINGS.MOSAIC_NUM
        self.useSSIM=DEFAULT_SETTINGS.useSSIM
        self.path_video='TBD'

        
        
        self.root_background_img=DEFAULT_SETTINGS.root_background_img #r"misc/gradient_blue.jpg"
        self.get_update_background_img()

        #self.root.config(menu=self.menubar)
        self.root.configure(bg=self.root_bg)
        self.drop_targets=None
        self.CWD=os.getcwd()
        self.not_checked_label=None
        self.checked_label_good=None
        self.checked_label_bad=None
        self.total_not_checked_label=None
        self.total_checked_label_good=None
        self.total_checked_label_bad=None

        self.annos_to_fix_label=None #os.listdir(self.path_Annotations_tofix)
        self.jpegs_to_fix_label=None #os.listdir(self.path_JPEGImages_tofix)
        self.jpegs_to_fix_bbox_lable=None #os.listdir(self.path_JPEGImages_tofix_bbox)  
        self.MOSAIC=None


        self.open_anno_label_var=tk.StringVar()
        self.open_anno_label_var.set(self.path_Annotations)

        self.open_anno_button=Button(self.root,image=self.icon_folder,command=partial(self.select_folder,self.path_Annotations,'Open Annotations Folder',self.open_anno_label_var),bg=self.root_bg,fg=self.root_fg)
        self.open_anno_button.grid(row=2,column=1,sticky='se')
        self.open_anno_note=tk.Label(self.root,text="1.a \n Annotations dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_anno_note.grid(row=3,column=1,sticky='ne')

        cmd_i=open_cmd+" '{}'".format(self.open_anno_label_var.get())
        self.open_anno_label=Button(self.root,textvariable=self.open_anno_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))

        self.open_anno_label.grid(row=2,column=2,columnspan=50,sticky='sw')

        self.open_jpeg_label_var=tk.StringVar()
        self.open_jpeg_label_var.set(self.path_JPEGImages)

        self.open_jpeg_button=Button(self.root,image=self.icon_folder,command=partial(self.select_folder,self.path_JPEGImages,'Open JPEGImages Folder',self.open_jpeg_label_var),bg=self.root_bg,fg=self.root_fg)
        self.open_jpeg_button.grid(row=4,column=1,sticky='se')
        self.open_jpeg_note=tk.Label(self.root,text="1.b \n JPEGImages dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_jpeg_note.grid(row=5,column=1,sticky='ne')

        self.open_video_label_var=tk.StringVar()
        self.open_video_label_var.set(self.path_video)
        self.open_video_button=Button(self.root,image=self.icon_folder,command=self.select_video,bg=self.root_bg,fg=self.root_fg)
        self.open_video_button.grid(row=2,column=31,sticky='se')
        self.create_video_button=Button(self.root,image=self.icon_video,command=self.create_JPEGImages_from_video,bg=self.root_bg,fg=self.root_fg)
        self.create_video_button.grid(row=2,column=30,sticky='se')
        self.open_video_note=tk.Label(self.root,text="1.c \n Create JPEGImages from Video File",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_video_note.grid(row=3,column=30,sticky='ne') 

        cmd_i=open_cmd+" '{}'".format(self.open_video_label_var.get())
        self.open_video_label=Button(self.root,textvariable=self.open_video_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_video_label.grid(row=2,column=32,columnspan=50,sticky='sw')    

        cmd_i=open_cmd+" '{}'".format(self.open_jpeg_label_var.get())
        self.open_jpeg_label=Button(self.root,textvariable=self.open_jpeg_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))

        self.open_jpeg_label.grid(row=4,column=2,columnspan=50,sticky='sw')
        if os.path.exists(self.path_Annotations) and os.path.exists(self.path_JPEGImages):
            self.MOSAIC=MOSAIC(self.path_JPEGImages,self.path_Annotations)
            if os.path.exists(self.MOSAIC.df_filename)==True:
                self.load_df_button=Button(self.root,image=self.icon_load,command=self.load_df,bg=self.root_bg,fg=self.root_fg)
                self.load_df_button.grid(row=8,column=1,sticky='se')
                self.load_note=tk.Label(self.root,text='3. \n Load df     ',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
                self.load_note.grid(row=9,column=1,sticky='ne')
                
            self.create_df_button=Button(self.root,image=self.icon_create,command=self.create_df,bg=self.root_bg,fg=self.root_fg)
            self.create_df_button.grid(row=6,column=1,sticky='se')
            self.create_note=tk.Label(self.root,text='2. \n Create df    ',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.create_note.grid(row=7,column=1,sticky='ne')
            try:
                self.open_originals_with_labelImg_note.destroy()
            except:
                pass
            try:
                self.open_originals_with_labelImg_button.destroy()
            except:
                pass
            self.open_originals_with_labelImg_button=Button(self.root,image=self.icon_labelImg,command=self.open_originals_with_labelImg_START,bg=self.root_bg,fg=self.root_fg)
            self.open_originals_with_labelImg_button.grid(row=2,column=29,sticky='se')
            self.open_originals_with_labelImg_note=tk.Label(self.root,text='12. \n Open originals w/ labelImg.py',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.open_originals_with_labelImg_note.grid(row=3,column=29,sticky='ne')


        self.save_settings_button=Button(self.root,image=self.icon_save_settings,command=self.save_settings,bg=self.root_bg,fg=self.root_fg)
        self.save_settings_button.grid(row=22,column=1,sticky='se')
        self.save_settings_note=tk.Label(self.root,text='Save Settings',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.save_settings_note.grid(row=23,column=1,sticky='ne')

    def play_video(self):
        self.open_video_label.destroy()
        cmd_i=open_cmd+" '{}'".format(self.open_video_label_var.get())
        self.open_video_label=Button(self.root,textvariable=self.open_video_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_video_label.grid(row=2,column=32,columnspan=50,sticky='sw')      

    def select_video(self,):
        
        filetypes=(('mp4','*.mp4'),('MP4','*.MP4'),('MOV','*.MOV'),('All files','*.*'))
        self.filename=fd.askopenfilename(title='Open a file',
                                    initialdir=self.CWD,
                                    filetypes=filetypes)
        if os.path.exists(self.filename):
            print('Yes {}'.format(self.filename))
        else:
            print('No {}'.format(self.filename))
        self.path_video=self.filename
        self.open_video_label_var.set(self.filename)
        if os.path.exists(self.path_video):
            pass
        self.play_video()

        showinfo(title='Selected File',
                 message=self.filename)
    def popup_fps(self):
        if _platform=='darwin':
            root_tk.withdraw()
        root_tk.title('Select the desired FPS from the dropdown')
        original_root_tk=root_tk
        self.w=popupWindowDropDown_FPS(root_tk)
        original_root_tk.wait_window(self.w.top)
        print(self.w.value)
        root_tk.title("MOSAIC Chip Sorter")
        if _platform=='darwin':
            original_root_tk.update()
            original_root_tk.deiconify()
        return self.w.value
    def create_JPEGImages_from_video(self):
        if os.path.exists(self.path_video):
            video_filename=os.path.join(os.path.dirname(self.path_video),os.path.basename(self.path_video).split('.')[0])
            if os.path.exists(video_filename)==False:
                os.makedirs(video_filename)
            else:
                shutil.move(video_filename,video_filename+'_backup_{}'.format(str(time.time()).split('.')[0]))
                os.makedirs(video_filename)
            video_filepath=os.path.join(os.path.dirname(self.path_video),video_filename)
            shutil.copy(self.path_video,video_filepath)
            self.path_video=os.path.join(video_filepath,os.path.basename(self.path_video))
            fps=self.popup_fps()
            new_anno_path,new_jpeg_path=create_imgs_from_video(self.path_Annotations,self.path_JPEGImages,path_movie=self.path_video,fps=fps)
            self.path_Annotations=new_anno_path
            self.path_JPEGImages=new_jpeg_path


            self.anno_selected=True
            self.open_anno_label_var.set(self.path_Annotations)
            self.open_anno_label.destroy()
            del self.open_anno_label
            cmd_i=open_cmd+" '{}'".format(self.open_anno_label_var.get())
            self.open_anno_label=Button(self.root,textvariable=self.open_anno_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
            #self.open_anno_label=tk.Label(self.root,textvariable=self.open_anno_label_var)
            self.open_anno_label.grid(row=2,column=2,columnspan=50,sticky='sw')
            print(self.path_Annotations)

            self.jpeg_selected=True
            self.open_jpeg_label_var.set(self.path_JPEGImages)
            self.open_jpeg_label.destroy()
            del self.open_jpeg_label
            cmd_i=open_cmd+" '{}'".format(self.open_jpeg_label_var.get())
            self.open_jpeg_label=Button(self.root,textvariable=self.open_jpeg_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
            #self.open_jpeg_label=tk.Label(self.root,textvariable=self.open_jpeg_label_var)
            self.open_jpeg_label.grid(row=4,column=2,columnspan=50,sticky='sw')
            print(self.path_JPEGImages)
            try:
                self.open_originals_with_labelImg_note.destroy()
            except:
                pass
            try:
                self.open_originals_with_labelImg_button.destroy()
            except:
                pass
            self.open_originals_with_labelImg_button=Button(self.root,image=self.icon_labelImg,command=self.open_originals_with_labelImg_START,bg=self.root_bg,fg=self.root_fg)
            self.open_originals_with_labelImg_button.grid(row=2,column=29,sticky='se')
            self.open_originals_with_labelImg_note=tk.Label(self.root,text='12. \n Open originals w/ labelImg.py',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.open_originals_with_labelImg_note.grid(row=3,column=29,sticky='ne')

    def create_move_Anno_JPEG(self):
        if os.path.basename(self.path_Annotations)!="Annotations":
            path_anno=os.path.join(self.path_Annotations,'Annotations')
            if os.path.exists(path_anno)==False:
                os.makedirs(path_anno)
            [shutil.move(os.path.join(self.path_Annotations,w),path_anno) for w in os.listdir(self.path_Annotations) if w.find('.xml')!=-1]
            self.path_Annotations=path_anno
        if os.path.basename(self.path_JPEGImages)!="JPEGImages":
            path_jpegs=os.path.join(self.path_JPEGImages,'JPEGImages')
            if os.path.exists(path_jpegs)==False:
                os.makedirs(path_jpegs)
            [shutil.move(os.path.join(self.path_JPEGImages,w),path_jpegs) for w in os.listdir(self.path_JPEGImages) if w.find('.jpg')!=-1]
            self.path_JPEGImages=path_jpegs
            
    def close(self,event):
        self.root.destroy()


    def save_settings(self):
        if os.path.exists('libs/DEFAULT_SETTINGS.py'):
            f=open('libs/DEFAULT_SETTINGS.py','r')
            f_read=f.readlines()
            f.close()
            f_new=[]
            for line in f_read:
                prefix_i=line.split('=')[0]
                try:
                    prefix_i_comb="self."+prefix_i
                    prefix_i_comb=prefix_i_comb.strip()
                    print(prefix_i_comb)
                    prefix_i_value=eval(prefix_i_comb)
                except:
                    prefix_i_comb="self.MOSAIC."+prefix_i
                    prefix_i_comb=prefix_i_comb.strip()
                    print(prefix_i_comb)
                    prefix_i_value=eval(prefix_i_comb)
                if line.split('=')[1].find("r'")!=-1:
                    prefix_i_value="r'"+prefix_i_value+"'"
                elif type(prefix_i_value).__name__.find('int')!=-1:
                    pass
                elif type(prefix_i_value).__name__.find('str')!=-1:
                    prefix_i_value="'"+prefix_i_value+"'"
                f_new.append(prefix_i+"="+str(prefix_i_value)+'\n')
            f=open('libs/SAVED_SETTINGS.py','w')
            wrote=[f.writelines(w) for w in f_new]
            f.close()

    def fix_with_labelImg(self):
        self.path_labelImg_image_dir=self.MOSAIC.path_JPEGImages_tofix
        self.path_labelImg_predefined_classes_file=os.path.join(self.MOSAIC.basePath,self.predefined_classes)
        f=open(self.path_labelImg_predefined_classes_file,'w')
        [f.writelines(str(w)+'\n') for w in list(self.MOSAIC.unique_labels.keys())]
        f.close()
        self.path_labelImg_save_dir=self.MOSAIC.path_Annotations_tofix
        if os.path.exists(self.path_labelImg):
            self.cmd_i='{} "{}" "{}" "{}" "{}"'.format(self.PYTHON_PATH ,self.path_labelImg,self.path_labelImg_image_dir,self.path_labelImg_predefined_classes_file,self.path_labelImg_save_dir)
            self.labelImg=Thread(target=self.run_cmd,args=(self.cmd_i,)).start()
        else:
            self.popup_text='Please provide a valid labelImg.py path. \n  Current path is: {}'.format(self.path_labelImg)
            self.fix_with_labelImg_error=tk.Label(self.root,text=self.popup_text, bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
            self.fix_with_labelImg_error.grid(row=15,column=2,sticky='s')
            self.fix_with_labelImg_error_button=Button(self.root,image=self.icon_folder, command=partial(self.select_file,self.path_labelImg),bg=self.root_bg,fg=self.root_fg)
            self.fix_with_labelImg_error_button.grid(row=14,column=2,sticky='s')

    def open_originals_with_labelImg(self):
        self.path_labelImg_image_dir=self.MOSAIC.path_JPEGImages
        self.path_labelImg_predefined_classes_file=os.path.join(self.MOSAIC.basePath,self.predefined_classes)
        f=open(self.path_labelImg_predefined_classes_file,'w')
        [f.writelines(str(w)+'\n') for w in list(self.MOSAIC.unique_labels.keys())]
        f.close()
        self.path_labelImg_save_dir=self.MOSAIC.path_Annotations
        if os.path.exists(self.path_labelImg):
            self.cmd_i='{} "{}" "{}" "{}" "{}"'.format(self.PYTHON_PATH ,self.path_labelImg,self.path_labelImg_image_dir,self.path_labelImg_predefined_classes_file,self.path_labelImg_save_dir)
            self.labelImg=Thread(target=self.run_cmd,args=(self.cmd_i,)).start()
        else:
            self.popup_text='Please provide a valid labelImg.py path. \n  Current path is: {}'.format(self.path_labelImg)
            self.fix_with_labelImg_error=tk.Label(self.root,text=self.popup_text, bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
            self.fix_with_labelImg_error.grid(row=23,column=32,sticky='s')
            self.fix_with_labelImg_error_button=Button(self.root,image=self.icon_folder, command=partial(self.select_file_START,self.path_labelImg),bg=self.root_bg,fg=self.root_fg)
            self.fix_with_labelImg_error_button.grid(row=22,column=32,sticky='s')

    def open_originals_with_labelImg_START(self):
        self.path_labelImg_image_dir=self.path_JPEGImages
        self.path_labelImg_predefined_classes_file=os.path.join(os.path.dirname(self.path_JPEGImages),self.predefined_classes)
        if os.path.exists(self.path_labelImg_predefined_classes_file)==False:
            f=open(self.path_labelImg_predefined_classes_file,'w')
            f.writelines('TBD \n')
            f.close()
        self.path_labelImg_save_dir=self.path_Annotations
        if os.path.exists(self.path_labelImg):
            self.cmd_i='{} "{}" "{}" "{}" "{}"'.format(self.PYTHON_PATH ,self.path_labelImg,self.path_labelImg_image_dir,self.path_labelImg_predefined_classes_file,self.path_labelImg_save_dir)
            self.labelImg=Thread(target=self.run_cmd,args=(self.cmd_i,)).start()
        else:
            self.popup_text='Please provide a valid labelImg.py path. \n  Current path is: {}'.format(self.path_labelImg)
            self.fix_with_labelImg_error=tk.Label(self.root,text=self.popup_text, bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
            self.fix_with_labelImg_error.grid(row=3,column=28,sticky='s')
            self.fix_with_labelImg_error_button=Button(self.root,image=self.icon_folder, command=partial(self.select_file_START,self.path_labelImg),bg=self.root_bg,fg=self.root_fg)
            self.fix_with_labelImg_error_button.grid(row=4,column=28,sticky='s')

    def pad(self,text_i,max_i):
        while len(text_i)!=max_i:
            text_i=text_i+" "
        return text_i
    def select_file(self,file_i):
        filetypes=(('py','*.py'),('All files','*.*'))
        self.filename=fd.askopenfilename(title='Open a file',
                                    initialdir=self.CWD,
                                    filetypes=filetypes)
        if file_i==self.path_labelImg:
            self.path_labelImg=self.filename
            if os.path.exists(self.path_labelImg):
                self.fix_with_labelImg_error.grid_forget()
                self.fix_with_labelImg_error_button.grid_forget()
                del self.fix_with_labelImg_error
                del self.fix_with_labelImg_error_button
            self.fix_with_labelImg()
        showinfo(title='Selected File',
                 message=self.filename)

    def select_file_START(self,file_i):
        filetypes=(('py','*.py'),('All files','*.*'))
        self.filename=fd.askopenfilename(title='Open a file',
                                    initialdir=self.CWD,
                                    filetypes=filetypes)
        if file_i==self.path_labelImg:
            self.path_labelImg=self.filename
            if os.path.exists(self.path_labelImg):
                self.fix_with_labelImg_error.grid_forget()
                self.fix_with_labelImg_error_button.grid_forget()
                del self.fix_with_labelImg_error
                del self.fix_with_labelImg_error_button
            self.open_originals_with_labelImg_START()
        showinfo(title='Selected File',
                 message=self.filename)


    def select_folder(self,folder_i,title_i,var_i=None):
        filetypes=(('All files','*.*'))
        if var_i:
            folder_i=var_i.get()
        if os.path.exists(folder_i):
            self.foldername=fd.askdirectory(title=title_i,
                                        initialdir=folder_i)
        else:
            self.foldername=fd.askdirectory(title=title_i)
        showinfo(title='Selected Folder',
                 message=self.foldername)
        folder_i=self.foldername
        if var_i==self.open_anno_label_var:
            self.anno_selected=True
            var_i.set(folder_i)
            self.open_anno_label.destroy()
            del self.open_anno_label
            cmd_i=open_cmd+" '{}'".format(self.open_anno_label_var.get())
            self.open_anno_label=Button(self.root,textvariable=self.open_anno_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
            #self.open_anno_label=tk.Label(self.root,textvariable=self.open_anno_label_var)
            self.open_anno_label.grid(row=2,column=2,columnspan=50,sticky='sw')
            self.path_Annotations=self.foldername
            print(self.path_Annotations)
        if var_i==self.open_jpeg_label_var:
            self.jpeg_selected=True
            var_i.set(folder_i)
            self.open_jpeg_label.destroy()
            del self.open_jpeg_label
            cmd_i=open_cmd+" '{}'".format(self.open_jpeg_label_var.get())
            self.open_jpeg_label=Button(self.root,textvariable=self.open_jpeg_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
            #self.open_jpeg_label=tk.Label(self.root,textvariable=self.open_jpeg_label_var)
            self.open_jpeg_label.grid(row=4,column=2,columnspan=50,sticky='sw')
            self.path_JPEGImages=self.foldername
            print(self.path_JPEGImages)
        if os.path.exists(self.path_JPEGImages) and os.path.exists(self.path_Annotations) and self.jpeg_selected and self.anno_selected:
            self.MOSAIC=MOSAIC(self.path_JPEGImages,self.path_Annotations)
            if os.path.exists(self.MOSAIC.df_filename)==True:
                self.load_df_button=Button(self.root,image=self.icon_load,command=self.load_df,bg=self.root_bg,fg=self.root_fg)
                self.load_df_button.grid(row=8,column=1,sticky='se')
                self.load_note=tk.Label(self.root,text='3. \n Load df     ',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
                self.load_note.grid(row=9,column=1,sticky='ne')
                
            self.create_df_button=Button(self.root,image=self.icon_create,command=self.create_df,bg=self.root_bg,fg=self.root_fg)
            self.create_df_button.grid(row=6,column=1,sticky='se')
            self.create_note=tk.Label(self.root,text='2. \n Create df    ',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.create_note.grid(row=7,column=1,sticky='ne')

    def run_cmd(self,cmd_i):
        os.system(cmd_i)
    def update_widget(self,widget):
        widget.grid_forget()
    def update_counts(self,target_i):
        if self.not_checked_label:
            self.not_checked_label.destroy()
        if self.checked_label_good:
            self.checked_label_good.destroy()
        if self.checked_label_bad:
            self.checked_label_bad.destroy()
        if self.total_not_checked_label:
            self.total_not_checked_label.destroy()
        if self.total_checked_label_good:
            self.total_checked_label_good.destroy()
        if self.total_checked_label_bad:
            self.total_checked_label_bad.destroy()
        self.MOSAIC.target_i=self.target_i.get()
        self.MOSAIC.update_checked()
        
        self.not_checked=self.MOSAIC.not_checked.get()
        self.not_checked="Not Checked '{}' Total = {}".format(self.target_i.get(),self.not_checked)
        self.not_checked_label=tk.Label(self.root,text=self.not_checked, bg=self.root_fg,fg=self.root_bg,font=("Arial", 10))
        self.not_checked_label.grid(row=10,column=4,sticky='w')

        self.checked_good=self.MOSAIC.checked_good.get()
        self.checked_good="Checked '{}' Good = {}".format(self.target_i.get(),self.checked_good)
        self.checked_label_good=tk.Label(self.root,text=self.checked_good, bg=self.root_fg,fg=self.root_bg,font=("Arial", 10))
        self.checked_label_good.grid(row=10,column=5,sticky='w')

        self.checked_bad=self.MOSAIC.checked_bad.get()
        self.checked_bad="Checked '{}' Bad = {}".format(self.target_i.get(),self.checked_bad)
        self.checked_label_bad=tk.Label(self.root,text=self.checked_bad, bg=self.root_fg,fg=self.root_bg,font=("Arial", 10))
        self.checked_label_bad.grid(row=10,column=6,sticky='w')

        self.total_not_checked=self.MOSAIC.total_not_checked.get()
        self.total_not_checked="Not Checked Total = {}".format(self.total_not_checked)
        self.total_not_checked_label=tk.Label(self.root,text=self.total_not_checked, bg=self.root_bg,fg=self.root_fg,font=("Arial", 10))
        self.total_not_checked_label.grid(row=9,column=4,sticky='w')

        self.total_checked_good=self.MOSAIC.total_checked_good.get()
        self.total_checked_good="Checked Good = {}".format(self.total_checked_good)
        self.total_checked_label_good=tk.Label(self.root,text=self.total_checked_good, bg=self.root_bg,fg=self.root_fg,font=("Arial", 10))
        self.total_checked_label_good.grid(row=9,column=5,sticky='w')

        self.total_checked_bad=self.MOSAIC.total_checked_bad.get()
        self.total_checked_bad="Checked Bad = {}".format(self.total_checked_bad)
        self.total_checked_label_bad=tk.Label(self.root,text=self.total_checked_bad, bg=self.root_bg,fg=self.root_fg,font=("Arial", 10))
        self.total_checked_label_bad.grid(row=9,column=6,sticky='w')
        
        self.MOSAIC.update_fix()
        self.annos_to_fix=len(self.MOSAIC.annos_to_fix)
        self.jpegs_to_fix=len(self.MOSAIC.jpegs_to_fix)
        self.jpegs_to_fix_bbox=len(self.MOSAIC.jpegs_to_fix_bbox) 

        self.annos_to_fix="Annos to fix = {}".format(self.annos_to_fix)
        cmd_i=open_cmd+" '{}'".format(self.MOSAIC.path_Annotations_tofix)
        self.annos_to_fix_label=Button(self.root,text=self.annos_to_fix, command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg,font=("Arial", 10))
        self.annos_to_fix_label.grid(row=11,column=4,sticky='w')

        self.jpegs_to_fix="JPEGS to fix = {}".format(self.jpegs_to_fix)
        cmd_i=open_cmd+" '{}'".format(self.MOSAIC.path_JPEGImages_tofix)
        self.jpegs_to_fix_label=Button(self.root,text=self.jpegs_to_fix, command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg,font=("Arial", 10))
        self.jpegs_to_fix_label.grid(row=11,column=5,sticky='w')

        self.jpegs_to_fix_bbox="JPEGS to fix bbox = {}".format(self.jpegs_to_fix_bbox)
        cmd_i=open_cmd+" '{}'".format(self.MOSAIC.path_JPEGImages_tofix_bbox)
        self.jpegs_to_fix_bbox_label=Button(self.root,text=self.jpegs_to_fix_bbox, command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg,font=("Arial", 10))
        self.jpegs_to_fix_bbox_label.grid(row=11,column=6,sticky='w')


        self.MOSAIC.MOSAIC_NUM=int(self.MOSAIC_NUM_VAR.get().strip())
        self.MOSAIC.DX=int(np.ceil(np.sqrt(self.MOSAIC.MOSAIC_NUM)))
        self.MOSAIC.DY=int(np.ceil(np.sqrt(self.MOSAIC.MOSAIC_NUM)))
        self.MOSAIC_NUM=self.MOSAIC.MOSAIC_NUM
        self.DX=self.MOSAIC.DX
        self.DY=self.MOSAIC.DY


    def update_targets(self):
        self.drop_targets.destroy()
        self.drop_targets=tk.OptionMenu(self.root,self.target_i,*self.options,command=self.update_counts)
        self.drop_targets.grid(row=10,column=2,sticky='s')
    def create_df(self):
        self.MOSAIC=MOSAIC(self.path_JPEGImages,self.path_Annotations)
        self.MOSAIC.create_df()
        if os.path.exists(self.MOSAIC.df_filename)==True:
            self.load_df_button=Button(self.root,image=self.icon_load,command=self.load_df,bg=self.root_bg,fg=self.root_fg)
            self.load_df_button.grid(row=8,column=1,sticky='se')
            self.load_note=tk.Label(self.root,text='3. \n Load df     ',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.load_note.grid(row=9,column=1,sticky='ne')
        self.load_df()
    def load_df(self):
        self.MOSAIC.load()
        self.options=list(self.MOSAIC.unique_labels.keys())
        self.target_i=tk.StringVar()
        self.target_i.set(self.options[0])
        self.MOSAIC_NUM_VAR=tk.StringVar()
        self.MOSAIC_NUM_VAR.set(self.MOSAIC_NUM)
        self.MOSAIC_NUM_entry=tk.Entry(self.root,textvariable=self.MOSAIC_NUM_VAR)
        self.MOSAIC_NUM_entry.grid(row=10,column=3,sticky='sw')
        self.MOSAIC_NUM_label=tk.Label(self.root,text='MOSAIC_NUM',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.MOSAIC_NUM_label.grid(row=11,column=3,sticky='nw')

        self.update_counts('na')
        
        if self.drop_targets==None:
            self.drop_targets=tk.OptionMenu(self.root,self.target_i,*self.options,command=self.update_counts)
            self.drop_targets.grid(row=10,column=2,sticky='s')
            self.drop_label=tk.Label(self.root,text='target',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
            self.drop_label.grid(row=11,column=2,sticky='nw')
        else:
            self.update_targets()
        self.analyze_target_button=Button(self.root,image=self.icon_analyze,command=self.analyze_target,bg=self.root_bg,fg=self.root_fg)
        self.analyze_target_button.grid(row=10,column=1,sticky='se')
        self.analyze_note=tk.Label(self.root,text='4. \n Analyze Mosaic',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.analyze_note.grid(row=11,column=1,sticky='ne')


        self.move_fix_button=Button(self.root,image=self.icon_move,command=self.move_fix,bg=self.root_bg,fg=self.root_fg)
        self.move_fix_button.grid(row=12,column=1,sticky='se')
        self.move_fix_note=tk.Label(self.root,text='5. \n Move Fix',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.move_fix_note.grid(row=13,column=1,sticky='ne')

        self.fix_with_labelImg_button=Button(self.root,image=self.icon_labelImg,command=self.fix_with_labelImg,bg=self.root_bg,fg=self.root_fg)
        self.fix_with_labelImg_button.grid(row=14,column=1,sticky='se')
        self.fix_labelImg_note=tk.Label(self.root,text='6. \n Fix w/ labelImg.py',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.fix_labelImg_note.grid(row=15,column=1,sticky='ne')


        self.merge_fix_button=Button(self.root,image=self.icon_merge,command=self.merge_fix,bg=self.root_bg,fg=self.root_fg)
        self.merge_fix_button.grid(row=16,column=1,sticky='se')
        self.merge_fix_note=tk.Label(self.root,text='7. \n Merge Fix',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.merge_fix_note.grid(row=17,column=1,sticky='ne')

        self.clear_fix_button=Button(self.root,image=self.icon_clear_fix,command=self.clear_fix,bg=self.root_bg,fg=self.root_fg)
        self.clear_fix_button.grid(row=18,column=1,sticky='se')
        self.clear_fix_note=tk.Label(self.root,text='8. \n Clear Fix',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.clear_fix_note.grid(row=19,column=1,sticky='ne')


        self.clear_checked_button=Button(self.root,image=self.icon_clear_checked,command=self.clear_checked,bg=self.root_bg,fg=self.root_fg)
        self.clear_checked_button.grid(row=20,column=1,sticky='se')
        self.clear_checked_note=tk.Label(self.root,text='9. \n Clear Checked',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.clear_checked_note.grid(row=21,column=1,sticky='ne')

        self.breakupDF_button=Button(self.root,image=self.icon_breakup,command=self.breakup_df,bg=self.root_bg,fg=self.root_fg)
        self.breakupDF_button.grid(row=20,column=30,sticky='se')
        self.breakupDF_note=tk.Label(self.root,text='10. \n Breakup df',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.breakupDF_note.grid(row=21,column=30,sticky='ne')

        self.UMAP_button=Button(self.root,image=self.icon_map,command=self.umap_update,bg=self.root_bg,fg=self.root_fg)
        self.UMAP_button.grid(row=20,column=31,sticky='se')
        self.UMAP_note=tk.Label(self.root,text='11. \n UMAP',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.UMAP_note.grid(row=21,column=31,sticky='ne')
        try:
            self.open_originals_with_labelImg_note.destroy()
        except:
            pass
        try:
            self.open_originals_with_labelImg_button.destroy()
        except:
            pass
        self.open_originals_with_labelImg_button=Button(self.root,image=self.icon_labelImg,command=self.open_originals_with_labelImg,bg=self.root_bg,fg=self.root_fg)
        self.open_originals_with_labelImg_button.grid(row=20,column=32,sticky='se')
        self.open_originals_with_labelImg_note=tk.Label(self.root,text='12. \n Open originals w/ labelImg.py',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.open_originals_with_labelImg_note.grid(row=21,column=32,sticky='ne')

        self.DX_MIN_VAR=tk.StringVar()
        self.DX_MIN_VAR.set(str(self.MOSAIC.DX_MIN))
        self.DX_MIN_entry=tk.Entry(self.root,textvariable=self.DX_MIN_VAR)
        self.DX_MIN_entry.grid(row=5,column=30,sticky='sw')
        self.DX_MIN_label=tk.Label(self.root,text='DX_MIN',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.DX_MIN_label.grid(row=6,column=30,sticky='nw')

        self.DX_MAX_VAR=tk.StringVar()
        self.DX_MAX_VAR.set(str(self.MOSAIC.DX_MAX))
        self.DX_MAX_entry=tk.Entry(self.root,textvariable=self.DX_MAX_VAR)
        self.DX_MAX_entry.grid(row=7,column=30,sticky='sw')
        self.DX_MAX_label=tk.Label(self.root,text='DX_MAX',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.DX_MAX_label.grid(row=8,column=30,sticky='nw')

        self.DY_MIN_VAR=tk.StringVar()
        self.DY_MIN_VAR.set(str(self.MOSAIC.DY_MIN))
        self.DY_MIN_entry=tk.Entry(self.root,textvariable=self.DY_MIN_VAR)
        self.DY_MIN_entry.grid(row=9,column=30,sticky='sw')
        self.DY_MIN_label=tk.Label(self.root,text='DY_MIN',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.DY_MIN_label.grid(row=10,column=30,sticky='nw')

        self.DY_MAX_VAR=tk.StringVar()
        self.DY_MAX_VAR.set(str(self.MOSAIC.DY_MAX))
        self.DY_MAX_entry=tk.Entry(self.root,textvariable=self.DY_MAX_VAR)
        self.DY_MAX_entry.grid(row=11,column=30,sticky='sw')
        self.DY_MAX_label=tk.Label(self.root,text='DY_MAX',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.DY_MAX_label.grid(row=12,column=30,sticky='nw')

        self.plot_DXDY_button=Button(self.root,image=self.icon_scatter,command=self.MOSAIC.plot_dx_dy,bg=self.root_bg,fg=self.root_fg)
        self.plot_DXDY_button.grid(row=5,column=31,sticky='se')
        self.plot_DXDY_note=tk.Label(self.root,text='13. \n Plot DX vs. DY',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.plot_DXDY_note.grid(row=6,column=31,sticky='ne')
        #self.plot_dx_dy(self)

        self.plot_DXDY_filter_button=Button(self.root,image=self.icon_filter,command=self.get_DXDY,bg=self.root_bg,fg=self.root_fg)
        self.plot_DXDY_filter_button.grid(row=7,column=31,sticky='se')
        self.plot_DXDY_filter_note=tk.Label(self.root,text='14. \n Filter DX,DY',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.plot_DXDY_filter_note.grid(row=8,column=31,sticky='ne')

        self.make_chips_button=Button(self.root,image=self.icon_create,command=self.make_chips,bg=self.root_bg,fg=self.root_fg)
        self.make_chips_button.grid(row=9,column=31,sticky='se')
        self.make_chips_note=tk.Label(self.root,text='15. \n Make Chips',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.make_chips_note.grid(row=10,column=31,sticky='ne')

        self.checkbutton_blanks()
        self.make_blank_chips_note=tk.Label(self.root,text='Make Blank Chips?',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.make_blank_chips_note.grid(row=10,column=32,sticky='ne')

        self.change_labels_button=Button(self.root,image=self.icon_create,command=self.change_labels,bg=self.root_bg,fg=self.root_fg)
        self.change_labels_button.grid(row=11,column=31,sticky='se')
        self.change_labels_note=tk.Label(self.root,text='16. \n Change Labels',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.change_labels_note.grid(row=12,column=31,sticky='ne') 

        self.remove_blanks_button=Button(self.root,image=self.icon_clear_fix,command=self.remove_blanks,bg=self.root_bg,fg=self.root_fg)
        self.remove_blanks_button.grid(row=13,column=31,sticky='se')
        self.remove_blanks_note=tk.Label(self.root,text='17. \n Remove Blanks',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.remove_blanks_note.grid(row=14,column=31,sticky='ne')    

        self.look_at_objects_button=Button(self.root,image=self.icon_analyze,command=self.look_at_objects,bg=self.root_bg,fg=self.root_fg)
        self.look_at_objects_button.grid(row=15,column=31,sticky='se')
        self.look_at_objects_note=tk.Label(self.root,text='18. \n Look at Objects',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.look_at_objects_note.grid(row=16,column=31,sticky='ne') 

        self.create_new_df_button=Button(self.root,image=self.icon_create,command=self.create_new_df,bg=self.root_bg,fg=self.root_fg)
        self.create_new_df_button.grid(row=17,column=31,sticky='se')
        self.create_new_df_note=tk.Label(self.root,text='19. \n Create New Dataset',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.create_new_df_note.grid(row=18,column=31,sticky='ne')       

        self.create_blackpatches_button=Button(self.root,image=self.icon_create,command=self.create_blackpatches,bg=self.root_bg,fg=self.root_fg)
        self.create_blackpatches_button.grid(row=13,column=32,sticky='se')
        self.create_blackpatches_note=tk.Label(self.root,text='20. \n Create Black Patches',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.create_blackpatches_note.grid(row=14,column=32,sticky='ne')   

        self.create_fakepatches_button=Button(self.root,image=self.icon_create,command=self.create_fakepatches,bg=self.root_bg,fg=self.root_fg)
        self.create_fakepatches_button.grid(row=17,column=32,sticky='se')
        self.create_fakepatches_note=tk.Label(self.root,text='21. \n Create New Dataset \n with Random Backgrounds',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.create_fakepatches_note.grid(row=18,column=32,sticky='ne') 

    def checkbutton_blanks(self):
        self.BLANK_boolean_var = tk.BooleanVar()
        self.style3=ttk.Style()
        self.style3.configure('Normal.TRadiobutton',
                             background='green',
                             foreground='black')
        self.option_yes = ttk.Radiobutton(self.root, text="Yes", style='Normal.TRadiobutton',variable=self.BLANK_boolean_var,
                                         value=True, command=self.callback_yes_no)
        self.option_no = ttk.Radiobutton(self.root, text="No", style='Normal.TRadiobutton', variable=self.BLANK_boolean_var,
                                        value=False, command=self.callback_yes_no)
        self.option_yes.grid(row=10, column=33,sticky='ne')
        self.option_no.grid(row=10, column=34,sticky='nw')
    def callback_yes_no(self):
        self.BLANK_boolean_var.set(self.BLANK_boolean_var.get())

    def get_DXDY(self):
        self.DX_MAX=int(self.DX_MAX_VAR.get())
        self.DX_MIN=int(self.DX_MIN_VAR.get())
        self.DY_MAX=int(self.DY_MAX_VAR.get())
        self.DY_MIN=int(self.DY_MIN_VAR.get())
        print(self.DX_MIN,self.DX_MAX,self.DY_MIN,self.DY_MAX)
        self.MOSAIC.filter_dxdy(self.DX_MIN,self.DX_MAX,self.DY_MIN,self.DY_MAX)
        self.load_df()
        self.update_counts('na')
    def breakup_df(self):
        self.MOSAIC.breakup_df()
        self.path_JPEGImages=self.MOSAIC.path_JPEGImages
        self.path_Annotations=self.MOSAIC.path_Annotations
        self.open_jpeg_label_var.set(self.path_JPEGImages)
        self.open_anno_label_var.set(self.path_Annotations)
        self.MOSAIC.load()
        self.update_counts('na')
    def clear_checked(self):
        self.MOSAIC.clear_checked()
        self.update_counts('na')
    def umap_update(self):
        self.MOSAIC.get_umap_output()
        self.MOSAIC.load()
        self.update_counts('na')
        self.load_df()
    def move_fix(self):
        self.MOSAIC.move_fix()
        self.update_counts('na')
    def merge_fix(self):
        self.MOSAIC.merge_fix()
        self.load_df()
        self.update_counts('na')
    def clear_fix(self):
        self.MOSAIC.clear_fix()
        self.update_counts('na')
    def analyze_target(self):
        self.update_targets()
        self.MOSAIC=MOSAIC(self.path_JPEGImages,self.path_Annotations)
        self.MOSAIC.MOSAIC_NUM=int(self.MOSAIC_NUM_VAR.get().strip())
        self.MOSAIC.DX=int(np.ceil(np.sqrt(self.MOSAIC.MOSAIC_NUM)))
        self.MOSAIC.DY=int(np.ceil(np.sqrt(self.MOSAIC.MOSAIC_NUM)))
        self.MOSAIC_NUM=self.MOSAIC.MOSAIC_NUM        
        self.DX=self.MOSAIC.DX
        self.DY=self.MOSAIC.DY
        self.MOSAIC.load()
        self.MOSAIC.look_at_target(self.target_i.get())
        self.update_counts('na')


    def make_chips(self):
        BLANK=self.BLANK_boolean_var.get()
        self.MOSAIC.chips(BLANK)
        self.update_counts('na')
    def create_new_df(self):
        self.MOSAIC.run_cmd=self.run_cmd
        self.MOSAIC.PYTHON_PATH=self.PYTHON_PATH
        self.MOSAIC.create_new_df()
        self.update_counts('na')
    def change_labels(self):
        self.MOSAIC.popup_changelabels()
        self.load_df()
        self.update_counts('na')
    def remove_blanks(self):
        self.MOSAIC.remove_blanks()
        self.load_df()
        self.update_counts('na')
    def create_blackpatches(self):
        self.MOSAIC.create_blackpatches()
        self.load_df()
        self.update_counts('na')
    def create_fakepatches(self):
        self.MOSAIC.create_fakepatches()
        self.load_df()
        self.update_counts('na')
    def look_at_objects(self):
        self.MOSAIC.look_at_objects()
        self.load_df()
        self.update_counts('na')
    def get_update_background_img(self):
        self.image=Image.open(self.root_background_img)
        self.image=self.image.resize((self.root_W,self.root_H),Image.ANTIALIAS)
        self.bg=ImageTk.PhotoImage(self.image)
        self.canvas=tk.Canvas(self.root,width=self.root_W,height=self.root_H)
        self.canvas.grid(row=0,column=0,columnspan=self.canvas_columnspan,rowspan=self.canvas_rowspan,sticky='nw')
        self.canvas.create_image(0,0,image=self.bg,anchor='nw')


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_JPEGImages",type=str,default='None',help='path JPEGImages')
    ap.add_argument("--path_Annotations",type=str,default='None',help='path Annotations')
    args = vars(ap.parse_args())
    path_JPEGImages=args['path_JPEGImages']
    path_Annotations=args['path_Annotations']
    my_app=App(root_tk,path_Annotations,path_JPEGImages)
    my_app.root.mainloop()


        
        


