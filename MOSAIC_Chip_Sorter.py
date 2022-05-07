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
from PIL import ImageDraw
from PIL import ImageFont
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter.tix import Balloon
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import umap 
from pprint import pprint
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
        self.basePath=self.path_Annotations.replace(self.path_Annotations.split('/')[-1],"")
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
            self.df_filename=os.path.join(self.basePath,"df_{}.pkl".format(self.basePath.split('/')[-2]))
        else:
            self.df_filename=df_filename
    def load(self):
        self.df=pd.read_pickle(self.df_filename)
        #df=pd.DataFrame(columns=['xmin','xmax','ymin','ymax','width','height','label_i','path_jpeg_i','path_anno_i'],dtype='object')
        self.int_cols=['xmin','xmax','ymin','ymax','width','height']
        for int_col in tqdm(self.int_cols):
            self.df[int_col]=self.df[int_col].astype(int)
        if 'checked' not in self.df.columns:
            self.df['checked']=self.df['path_anno_i'].copy()
            self.df['checked']=''  
        self.unique_labels={w:i for i,w in enumerate(self.df['label_i'].unique())}
        self.clear_fix_bad()
    def draw(self):
        self.df_fix=self.df[self.df['checked']=='bad'].reset_index().drop('index',axis=1)
        self.unique_labels={w:i for i,w in enumerate(self.df['label_i'].unique())}
        for anno,jpg in zip(self.df_fix.path_anno_i,self.df_fix.path_jpeg_i):
            anno_i=anno.split('/')[-1]
            jpg_i=jpg.split('/')[-1]
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
            anno_i=anno.split('/')[-1]
            jpg_i=jpg.split('/')[-1]
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
            anno_i=anno.split('/')[-1]
            jpg_i=jpg.split('/')[-1]
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
                self.df.at[i,'xmin']=xmin
                self.df.at[i,'xmax']=xmax
                self.df.at[i,'ymin']=ymin
                self.df.at[i,'ymax']=ymax
                self.df.at[i,'width']=width_i
                self.df.at[i,'height']=height_i
                self.df.at[i,'label_i']=label
                self.df.at[i,'path_jpeg_i']=path_jpeg_i.replace('_tofix','')
                self.df.at[i,'path_anno_i']=path_anno_i.replace('_tofix','')
                self.df.at[i,'checked']=''
                i+=1
        self.df.to_pickle(self.df_filename)
    def create_df(self):
        self.Annotations_list=list(os.listdir(self.path_Annotations))
        self.Annotations=[os.path.join(self.path_Annotations,Anno) for Anno in self.Annotations_list if Anno.find(self.XML_EXT)!=-1]
        self.JPEGs_list=list(os.listdir(self.path_JPEGImages))
        self.JPEGs=[os.path.join(self.path_JPEGImages,Anno.split(self.XML_EXT)[0]+self.JPG_EXT) for Anno in self.Annotations_list if Anno.split(self.XML_EXT)[0]+self.JPG_EXT in self.JPEGs_list]
        assert len(self.JPEGs)==len(self.Annotations) 

        self.df=pd.DataFrame(columns=['xmin','xmax','ymin','ymax','width','height','label_i','path_jpeg_i','path_anno_i'],dtype='object')
        i=0
        for path_anno_i,path_jpeg_i in tqdm(zip(self.Annotations,self.JPEGs)):
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
                self.df.at[i,'xmin']=xmin
                self.df.at[i,'xmax']=xmax
                self.df.at[i,'ymin']=ymin
                self.df.at[i,'ymax']=ymax
                self.df.at[i,'width']=width_i
                self.df.at[i,'height']=height_i
                self.df.at[i,'label_i']=label
                self.df.at[i,'path_jpeg_i']=path_jpeg_i
                self.df.at[i,'path_anno_i']=path_anno_i
                i+=1
        self.df.to_pickle(self.df_filename)
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
            self.go_to_next=True
        if event.key=='q' or event.key=='escape':
            plt.close('all')
            self.close_window=True
        self.df.to_pickle(self.df_filename)
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
                else:
                    self.df.at[index_bad,'checked']="bad" #resets
                    self.title_list[j].set_text('{} = BAD'.format(self.dic[j]))
                print(self.df.loc[index_bad])
                plt.show()
                #plt.pause(1e-3)
                break
        self.df.to_pickle(self.df_filename)
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
            self.inspect_mosaic=False
            self.run_selection=None
            #self.close_window_mosaic=True
        else:
            #plt.close('all')
            if event.key=='n':
                plt.close('all')
                self.go_to_next=True
                self.selection_list={}
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
                            for ii,line in enumerate(f_read):
                                if line.find(label_k)!=-1:
                                    print('found label_k',label_k)
                                    combo_i=f_read[ii-1:]
                                    combo_i="".join([w for w in combo_i])
                                    combo_i=combo_i.split('<object>')
                                    if combo_i[1].find(xmin)!=-1 and combo_i[1].find(xmax)!=-1 and combo_i[1].find(ymin)!=-1 and combo_i[1].find(ymax)!=-1:
                                        start_line=ii-1
                                        for jj,line_j in enumerate(f_read[ii:]):
                                            if line_j.find('</object>')!=-1:
                                                end_line=jj+ii
                                        f_new.append(line)
                                    else:
                                        f_new.append(line)
                                else:
                                    f_new.append(line)
                            f_new=f_new[:start_line]+f_new[end_line+1:]
                            f=open(path_anno_bad,'w')
                            [f.writelines(w) for w in f_new]
                            f.close()  
                            self.df.to_pickle(self.df_filename)
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
            print('event.inaxes',event.inaxes)   
            print('x1:',self.ex)
            print('y1:',self.ey)
            self.ex2=event.xdata
            self.ey2=event.ydata
            print('x2:',self.ex2)
            print('y2:',self.ey2)    

        #self.df.at[i,'distance']=np.sqrt(self.dx**2+self.dy**2)
        #self.df_to_fix_i=self.df.sort_values(by='distance',ascending=True).copy()
    def onclick_show(self,event):
        #global df_i,axes_list,df,title_list,img_list
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
             ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        print('event.inaxes',event.inaxes)
        try:
            cv2.destroyWindow('Selected Image.  Press "f" to fix.  Press "q" to quit.')
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
            text_labels='Press "d" to delete annotation for this object.\n\n'
            for label_j,int_i in self.label_dic.items():
                text_labels+='Press "{}" to change label to "{}"\n'.format(int_i,label_j)
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



    def on_key_show(self,event):
        print('you pressed', event.key, event.xdata, event.ydata)
        if event.key=='n' and self.inspect_mosaic==False:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        if event.key=='q' or event.key=='escape':
            plt.close('all')
            try:
                cv2.destroyAllWindows()
            except:
                pass
            self.inspect_mosaic=False
            self.close_window_mosaic=True
            self.run_selection=None
        if event.key=='f':
            print('fixing it')
            self.df.at[self.index_bad,'checked']="bad"
            self.df.to_pickle(self.df_filename)
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
            print(self.df_sample)
            self.inspect_mosaic=True
            print('looking at selection')
            self.run_selection=self.look_at_selection()
            self.inspect_mosaic=False
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
                for ii,line in enumerate(f_read):
                    if line.find(label_k)!=-1:
                        print('found label_k',label_k)
                        combo_i=f_read[ii-1:]
                        combo_i="".join([w for w in combo_i])
                        combo_i=combo_i.split('<object>')
                        if combo_i[1].find(xmin)!=-1 and combo_i[1].find(xmax)!=-1 and combo_i[1].find(ymin)!=-1 and combo_i[1].find(ymax)!=-1:
                            start_line=ii-1
                            for jj,line_j in enumerate(f_read[ii:]):
                                if line_j.find('</object>')!=-1:
                                    end_line=jj+ii
                            f_new.append(line)
                        else:
                            f_new.append(line)
                    else:
                        f_new.append(line)
                f_new=f_new[:start_line]+f_new[end_line+1:]
                f=open(path_anno_bad,'w')
                [f.writelines(w) for w in f_new]
                f.close()  
                self.df.to_pickle(self.df_filename)
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

        print(len(self.df_i))
        
        for k in tqdm(range(1+int(np.ceil(len(self.df_i)//self.MOSAIC_NUM)))):
            print('self.close_window_mosaic==',self.close_window_mosaic)
            if self.close_window_mosaic==True:
                self.close_window_mosaic=False
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
                  
                self.axes_list.append(self.fig_i.add_subplot(self.DX,self.DY,i+1-self.start))
                plt.subplots_adjust(wspace=0.2,hspace=0.5)
                
                self.img_list.append(self.chip_square_i)
                plt.imshow(self.chip_square_i)
                plt.axis('off')

                self.title_list.append(plt.title(self.df_i.iloc[i].label_i,fontsize='5',color='blue'))
                if i==len(self.df_i):
                    break
            plt.show()
            #self.df.to_pickle(self.df_filename) #TBD

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
            self.chip_i=self.jpg_i[self.ymin_i:self.ymax_i,self.xmin_i:self.xmax_i,:]
            self.chip_square_i=Image.fromarray(self.chip_i)              
            self.chip_square_i=self.chip_square_i.resize((self.W,self.H),Image.ANTIALIAS)
            self.chip_square_i=np.array(self.chip_square_i)
            self.df.at[i,'umap_input']=self.chip_square_i
        self.df.to_pickle(self.df_filename)
    def breakup_df(self):
        print(len(self.df))
        self.SPLIT_NUM=1000
        prefix_uniques=[w.split('/')[-1].split('.')[0] for w in self.df['path_anno_i'].unique()]
        if len(self.df)<self.SPLIT_NUM:
            self.SPLIT_NUM=len(self.df)
        count=0
        for i,(xml_i,jpeg_i) in tqdm(enumerate(zip(self.df['path_anno_i'],self.df['path_jpeg_i']))):
            if count%self.SPLIT_NUM==0:
                xml_list_i=[]
                jpeg_list_i=[]
                start_i=count
                end_i=start_i+self.SPLIT_NUM-1
                prefix_i=xml_i.split('/')[-1]
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
                df_i_filename=os.path.join(new_path_i,'df_{}.pkl'.format(new_path_i.split('/')[-1]))
                df_i.to_pickle(df_i_filename)
            count+=1
    def get_umap_output(self):
        self.df=self.df.dropna(axis=1)
        if 'emb_X' not in self.df.columns:
            if 'umap_input' not in self.df.columns:
                self.df['umap_input']=self.df['path_anno_i'].copy()
                self.df['umap_input']=''  
                self.get_umap_input()
            self.reducer=umap.UMAP(random_state=42)
            self.flattened=[np.array(w).flatten() for w in self.df.umap_input]
            self.reducer.fit(self.flattened)
            self.embedding=self.reducer.transform(self.flattened)
            self.df['emb_X']=self.df['label_i'].copy()
            self.df['emb_Y']=self.df['label_i'].copy()
            self.df['emb_X']=[w for w in self.embedding[:,0]]
            self.df['emb_Y']=[w for w in self.embedding[:,1]]
            self.df['emb_X']=self.df['emb_X'].astype(np.float16)
            self.df['emb_Y']=self.df['emb_Y'].astype(np.float16)
            self.label_dic={}
            i=0
            for label in self.df['label_i'].unique():
                if label not in self.label_dic.keys():
                    self.label_dic[label]=i
                    i+=1

            self.df['label_i_int']=self.df['label_i'].copy()
            self.df['label_i_int']=[self.label_dic[w] for w in self.df['label_i']]
            self.df['label_dist']=self.df['emb_X'].copy()
            for row in tqdm(range(len(self.df))):
                label_i=self.df.iloc[row].label_i
                avg_X_i=self.df[self.df.label_i==label_i]['emb_X'].mean()
                avg_Y_i=self.df[self.df.label_i==label_i]['emb_Y'].mean()
                X_i=self.df.iloc[row].emb_X
                Y_i=self.df.iloc[row].emb_Y
                d_X_i=X_i-avg_X_i
                d_Y_i=Y_i-avg_Y_i
                dist_i=np.sqrt(d_X_i**2+d_Y_i**2)
                self.df.at[row,'label_dist']=dist_i
        i=0
        self.label_dic={}
        for label in self.df['label_i'].unique():
            if label not in self.label_dic.keys():
                self.label_dic[label]=i
                i+=1
        self.rev_label_dic={v:k for k,v in self.label_dic.items()}
        self.df.to_pickle(self.df_filename)
        self.fig_j=plt.figure(figsize=(self.FIGSIZE_W,self.FIGSIZE_H),num='UMAP.  "Double Click" to inspect.  Press "q" to quit.    Press "m" for MOSAIC.')
        self.fig_j.set_size_inches((self.FIGSIZE_INCH_W, self.FIGSIZE_INCH_W))
        self.cidj = self.fig_j.canvas.mpl_connect('button_press_event', self.onclick_show)
        self.cidkj = self.fig_j.canvas.mpl_connect('key_press_event', self.on_key_show)
        self.cidkjm = self.fig_j.canvas.mpl_connect('button_release_event',self.release_show)
        self.close_window_mosaic=False

        plt.rcParams['axes.facecolor'] = 'gray'
        plt.grid(c='white')
        text_labels='Press "d" to delete annotation for this object.\n\n'
        
        for label_j,int_i in self.label_dic.items():
            text_labels+='Press "{}" to change label to "{}"\n'.format(int_i,label_j)
        plt.xlabel(text_labels)
        plt.scatter(self.df['emb_X'],self.df['emb_Y'],c=self.df['label_i_int'],cmap='Spectral',s=5)
        self.ex=min(self.df['emb_X'])
        self.ex2=max(self.df['emb_X'])
        self.ey=min(self.df['emb_Y'])
        self.ey2=max(self.df['emb_Y'])
        self.gca=plt.gca()
        self.gca.set_aspect('equal','datalim')
        plt.colorbar(boundaries=np.arange(len(self.df['label_i_int'].unique())+1)-0.5).set_ticks(np.arange(len(self.df['label_i_int'].unique())))
        plt.tight_layout()
        plt.show()


class App:
    def __init__(self,root_tk):
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
        self.path_JPEGImages=DEFAULT_SETTINGS.path_JPEGImages #r'/Volumes/One Touch/Images_gdrive/Drone_Images/Training/JPEGImages'
        self.path_Annotations=DEFAULT_SETTINGS.path_Annotations #r'/Volumes/One Touch/Images_gdrive/Drone_Images/Training/Annotations'
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


        self.save_settings_button=Button(self.root,image=self.icon_save_settings,command=self.save_settings,bg=self.root_bg,fg=self.root_fg)
        self.save_settings_button.grid(row=22,column=1,sticky='se')
        self.save_settings_note=tk.Label(self.root,text='Save Settings',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.save_settings_note.grid(row=23,column=1,sticky='ne')

    def create_move_Anno_JPEG(self):
        if self.path_Annotations.split('/')[-1]!="Annotations":
            path_anno=os.path.join(self.path_Annotations,'Annotations')
            if os.path.exists(path_anno)==False:
                os.makedirs(path_anno)
            [shutil.move(os.path.join(self.path_Annotations,w),path_anno) for w in os.listdir(self.path_Annotations) if w.find('.xml')!=-1]
            self.path_Annotations=path_anno
        if self.path_JPEGImages.split('/')[-1]!="JPEGImages":
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
            self.fix_with_labelImg_error_button=Button(self.root,image=self.icon_folder, command=partial(self.select_file,self.path_labelImg),bg=self.root_bg,fg=self.root_fg)
            self.fix_with_labelImg_error_button.grid(row=22,column=32,sticky='s')


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

        self.open_originals_with_labelImg_button=Button(self.root,image=self.icon_labelImg,command=self.open_originals_with_labelImg,bg=self.root_bg,fg=self.root_fg)
        self.open_originals_with_labelImg_button.grid(row=20,column=32,sticky='se')
        self.open_originals_with_labelImg_note=tk.Label(self.root,text='12. \n Open originals w/ labelImg.py',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.open_originals_with_labelImg_note.grid(row=21,column=32,sticky='ne')

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
        self.update_counts('na')
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
    def get_update_background_img(self):
        self.image=Image.open(self.root_background_img)
        self.image=self.image.resize((self.root_W,self.root_H),Image.ANTIALIAS)
        self.bg=ImageTk.PhotoImage(self.image)
        self.canvas=tk.Canvas(self.root,width=self.root_W,height=self.root_H)
        self.canvas.grid(row=0,column=0,columnspan=self.canvas_columnspan,rowspan=self.canvas_rowspan,sticky='nw')
        self.canvas.create_image(0,0,image=self.bg,anchor='nw')


if __name__=='__main__':
    my_app=App(root_tk)
    my_app.root.mainloop()


        
        


