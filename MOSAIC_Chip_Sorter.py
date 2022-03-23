'''
MOSAIC_Chip_Sorter
========
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

'''
import sys
from sys import platform as _platform
import pandas as pd
from tqdm import tqdm
import os
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
try:
    from libs import SAVED_SETTINGS as DEFAULT_SETTINGS
except:
    from libs import DEFAULT_SETTINGS 

if _platform=='darwin':
    import tkmacosx
    from tkmacosx import Button as Button
    open_cmd='open'
else:
    from tkinter import Button as Button
    if _platform.lower().find('linux')!=-1:
        open_cmd='xdg-open'
    else:
        open_cmd='start'
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
                    #self.title_list[j].set_text(str(self.dic[j]))
                else:
                    self.df.at[index_bad,'checked']="bad" #resets
                    #self.title_list[j].set_text('{} = BAD'.format(self.dic[j]))
                print(self.df.loc[index_bad])
                plt.show()
                plt.pause(1e-3)
                break
        self.df.to_pickle(self.df_filename)
    def look_at_target(self,target_i):
        self.target_i=target_i
        self.df_i=self.df[(self.df['label_i']==self.target_i) & (self.df['checked']=='')]
        print(len(self.df_i))
        
        for k in tqdm(range(1+int(np.ceil(len(self.df_i)//self.MOSAIC_NUM)))):

            if self.close_window==True:
                break
            self.go_to_next=False
            self.axes_list=[]
            #self.title_list=[]
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
                try:
                    self.chip_i=self.jpg_i[self.ymin_i-5:self.ymin_i+self.longest+5,self.xmin_i-5:self.xmin_i+self.longest+5,:]
                    self.chip_square_i=Image.fromarray(self.chip_i)
                except:
                    try:
                        self.chip_i=self.jpg_i[self.ymin_i:self.ymin_i+self.longest,self.xmin_i:self.xmin_i+self.longest,:]
                        self.chip_square_i=Image.fromarray(self.chip_i)
                    except:
                        self.chip_i=self.jpg_i[self.ymin_i:self.ymax_i,self.xmin_i:self.xmax_i,:]
                        self.chip_square_i=Image.fromarray(self.chip_i)                
                self.chip_square_i=self.chip_square_i.resize((self.W,self.H),Image.ANTIALIAS)
                self.chip_square_i=np.array(self.chip_square_i)
                self.df.at[self.df_i.iloc[i].name,'checked']='good'
                self.axes_list.append(self.fig_i.add_subplot(self.DX,self.DY,i+1-self.start))
                plt.subplots_adjust(wspace=0.1,hspace=0.1)
                
                self.img_list.append(self.chip_square_i)
                plt.imshow(self.chip_square_i)
                plt.axis('off')
                #self.title_list.append(plt.title(i,fontsize='5',color='red'))
                if i==len(self.df_i):
                    break
            plt.show()
            #while self.go_to_next==False and self.close_window==False:
            #    time.sleep(0.1)
            #    pass
            self.df.to_pickle(self.df_filename)

class App:
    def __init__(self):
        self.root=tk.Tk()
        self.root.bind('<Escape>',self.close)
        self.root.wm_iconphoto(False,ImageTk.PhotoImage(Image.open("resources/icons/appM.png")))
        self.icon_folder=ImageTk.PhotoImage(Image.open("resources/icons/file.png"))
        self.icon_load=ImageTk.PhotoImage(Image.open('resources/icons/load.png'))
        self.icon_create=ImageTk.PhotoImage(Image.open('resources/icons/create.png'))
        self.icon_analyze=ImageTk.PhotoImage(Image.open('resources/icons/analyze.png'))
        self.icon_move=ImageTk.PhotoImage(Image.open('resources/icons/move.png'))
        self.icon_labelImg=ImageTk.PhotoImage(Image.open('resources/icons/labelImg.png'))
        self.icon_merge=ImageTk.PhotoImage(Image.open('resources/icons/merge.png'))
        self.icon_clear_fix=ImageTk.PhotoImage(Image.open('resources/icons/clear.png'))
        self.icon_clear_checked=ImageTk.PhotoImage(Image.open('resources/icons/clear_checked.png'))
        self.icon_save_settings=ImageTk.PhotoImage(Image.open('resources/icons/save.png'))
        self.path_JPEGImages=DEFAULT_SETTINGS.path_JPEGImages #r'/Volumes/One Touch/Images_gdrive/Drone_Images/Training/JPEGImages'
        self.path_Annotations=DEFAULT_SETTINGS.path_Annotations #r'/Volumes/One Touch/Images_gdrive/Drone_Images/Training/Annotations'
        self.path_labelImg=DEFAULT_SETTINGS.path_labelImg #r'/Volumes/One Touch/labelImg-Custom/labelImg.py'
        self.jpeg_selected=False #after  user has opened from folder dialog the jpeg folder, this returns True
        self.anno_selected=False #after user has opened from folder dialog the annotation folder, this returns True
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

        ######################################### Left Button ##############
        self.left_button_texts=[
        'Step 1a. Select Annotations Folder ',
        'Step 1b. Select JPEGImages Folder',
        'Step 2. Create DF',
        'Step 3. Load DF',
        'Step 4.  Analyze Target',
        'Step 5.  move_fix',
        'Step 6.  Fix with labelImg',
        'Step 7.  merge_fix',
        'Step 8.  clear_fix',
        'Step 9.  clear_checked',
]
        #self.open_anno_label=None
        self.open_anno_label_var=tk.StringVar()
        self.open_anno_label_var.set(self.path_Annotations)
        #text=self.left_button_texts[0]
        self.open_anno_button=Button(self.root,image=self.icon_folder,command=partial(self.select_folder,self.path_Annotations,'Open Annotations Folder',self.open_anno_label_var),bg=self.root_bg,fg=self.root_fg)
        self.open_anno_button.grid(row=2,column=1,sticky='se')
        self.open_anno_note=tk.Label(self.root,text="1.a \n Annotations dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_anno_note.grid(row=3,column=1,sticky='ne')

        cmd_i=open_cmd+" '{}'".format(self.open_anno_label_var.get())
        self.open_anno_label=Button(self.root,textvariable=self.open_anno_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        #self.open_anno_label=tk.Label(self.root,textvariable=self.open_anno_label_var)
        self.open_anno_label.grid(row=2,column=2,columnspan=50,sticky='sw')

        #self.open_jpeg_label=None
        self.open_jpeg_label_var=tk.StringVar()
        self.open_jpeg_label_var.set(self.path_JPEGImages)
        #text=self.left_button_texts[1]
        self.open_jpeg_button=Button(self.root,image=self.icon_folder,command=partial(self.select_folder,self.path_JPEGImages,'Open JPEGImages Folder',self.open_jpeg_label_var),bg=self.root_bg,fg=self.root_fg)
        self.open_jpeg_button.grid(row=4,column=1,sticky='se')
        self.open_jpeg_note=tk.Label(self.root,text="1.b \n JPEGImages dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_jpeg_note.grid(row=5,column=1,sticky='ne')

        cmd_i=open_cmd+" '{}'".format(self.open_jpeg_label_var.get())
        self.open_jpeg_label=Button(self.root,textvariable=self.open_jpeg_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        #self.open_jpeg_label=tk.Label(self.root,textvariable=self.open_jpeg_label_var)
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
    def clear_checked(self):
        self.MOSAIC.clear_checked()
        self.update_counts('na')
    def move_fix(self):
        self.MOSAIC.move_fix()
        self.update_counts('na')
    def merge_fix(self):
        self.MOSAIC.merge_fix()
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
    my_app=App()
    my_app.root.mainloop()


        
        


