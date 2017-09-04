#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 12:17:37 2017

@author: acosi
"""

import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import re
from stemming.porter2 import stem
import pickle as pkl
import sys
from html_class import MLStripper
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--bad_folders", default=None, nargs='+')
parser.add_argument("-g", "--good_folders", default=None, nargs='+')

# def create_y_vector(bad_folder,good_folder):
#     bad_files = np.array([[x,0] for x in os.listdir(bad_folder) if os.path.isfile(os.path.join(bad_folder,x))])
#     good_files = np.array([[x,1] for x in os.listdir(good_folder) if os.path.isfile(os.path.join(good_folder,x))])
#     all_files = np.concatenate((bad_files,good_files))
#     x_train, x_test, y_train, y_test = train_test_split(all_files[:,0], all_files[:,1],test_size=0.3,stratify=all_files[:,1])
#     return x_train, x_test, y_train, y_test
def replace_number(st,reg):
    return reg.sub('number',st)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def process_email(email):
    sub_num = re.compile('\d{1,100}')
    space_reg = re.compile('\s')
    non_char_reg = re.compile('(?:[^\w\s@]|_)')
    with open(email,'br') as f:
        content = f.read().decode('utf-8', 'ignore').lower()
    content = non_char_reg.sub('',strip_tags(content.replace('$','dollar').replace('£','pound')))
    words_to_process = content.split()
    words_to_process = ['httpaddr' if 'http' in x else x for x in words_to_process]
    words_to_process = [replace_number(x, sub_num) for x in  words_to_process]
    words_to_process = ['emailaddr' if '@' in x else x for x in words_to_process]
    words_to_process = map(lambda x: space_reg.sub('',x), words_to_process)
    words_to_process = map(lambda x: stem(x), words_to_process)
    #words_to_process = map(lambda x: x.encode('utf-8', 'ignore'), words_to_process)
    return list(words_to_process)

def map_to_index(email,ind_to_word):
    with open(email,'br') as f:
        content = f.read().decode('utf-8', 'ignore')
    indices = []
    for word in content.split():
        if word in ind_to_word:
            indices += [ind_to_word[word]]
    return indices

def create_features_frequencies(indexed_email,ind_to_freq):
    arr_of_features = np.zeros(len(ind_to_word))
    with open(indexed_email) as f:
        content = f.read().split()
    for ind,count in np.unique(np.array(content)):
        if ind in ind_to_freq:
            arr_of_features[int(ind)] = float(count/ind_to_freq[ind])
    return arr_of_features.reshape((1,-1))

def create_features(indexed_email,ind_to_word):
    arr_of_features = np.zeros(len(ind_to_word))
    with open(indexed_email) as f:
        content = f.read().split()
    for n in content:
        if n in ind_to_word:
            arr_of_features[int(n)] = 1
    return arr_of_features.reshape((1,-1))

def pre_process_split(bad_folders,good_folders,ind_to_word=None):
    folders = bad_folders+good_folders
    bad_files = []
    good_files = []
    for inp_folder in folders:
        os.makedirs(os.path.join(inp_folder,'processed'))
    for bad_folder in bad_folders:
        bad_files += [[os.path.join(bad_folder,x),0] for x in os.listdir(bad_folder) if os.path.isfile(os.path.join(bad_folder,x))]
    for good_folder in good_folders:
        good_files += [[os.path.join(good_folder,x),1] for x in os.listdir(good_folder) if os.path.isfile(os.path.join(good_folder,x))]
    all_files = np.concatenate((np.array(bad_files),np.array(good_files)))

    df = pd.DataFrame(all_files,columns=['file_identifier','dependent_variable'])
    df = df.set_index('file_identifier')

    x_train, x_test, y_train, y_test = train_test_split(all_files[:,0], all_files[:,1],
                                                        test_size=0.3,stratify=all_files[:,1])
    ind_to_word = create_mapping(x_train,x_test,ind_to_word)
    return x_train, x_test, y_train, y_test, df, ind_to_word

def process_and_features(x_train,x_test,df,ind_to_word,folders):
    for inp_folder in folders:
        os.makedirs(os.path.join(inp_folder,'final'))
    words = list(ind_to_word.keys())
    df_fin = pd.DataFrame(index=df.index, columns=['dependent_variable']+words)
    df_fin.dependent_variable = df.dependent_variable
    first_word = words[0]
    for y in df.index:
        inp_fl = os.path.join(os.path.dirname(y),'processed','{}'.format(os.path.basename(y)))
        proc_out = map_to_index(inp_fl,ind_to_word)
        with open(inp_fl.replace('processed','final'),'w') as w:
            w.write(' '.join(proc_out))
        df_fin.loc[y,first_word:] = create_features(os.path.join(os.path.dirname(y),'final/{}'.format(os.path.basename(y))), {ind_to_word[x]:x for x in ind_to_word})
    df_train, df_test = df_fin.loc[x_train], df_fin.loc[x_test]
    with open('train_test_targets.pkl','wb') as wr:
        pkl.dump({'X_train':df_train, 'X_test':df_test},wr)
    with open('X_and_y.pkl','wb') as f:
        pkl.dump({'targets':df_train.append(df_test).iloc[:,0],
        'features':df_train.append(df_test).iloc[:,1:]},f)
    return df_train, df_test, ind_to_word

def create_mapping(train_files,test_files,ind_to_word=None,frequencies=False):
    '''Create mapping and pre_process'''
    res = []
    for y in np.concatenate((train_files,test_files)):
        proc_out = process_email(y)
        with open(os.path.join(os.path.dirname(y),'processed',
                               '{}'.format(os.path.basename(y))),'w') as w:
            w.write(' '.join(proc_out))
        if not ind_to_word:
            if y in train_files:
                res += proc_out
    if frequencies:
        print('counting frequencies')
        unique,counts = np.unique(np.array(res),return_counts=True)
        inds = counts > 140
        v = dict(zip(unique[inds], (counts[inds]/train_files.shape[0]).astype(str)))
        for i,te in enumerate(v):
            with open('voc_freq.txt','a') as w:
                w.write(str(i)+' '+te+'\n')
        ind_to_word = dict()
        with open('voc_freq.txt') as f:
            for line in f:
                ind, word = line.strip().split()
                ind_to_word[word] = ind
    if not ind_to_word:
        print('counting terms')
        unique,counts = np.unique(np.array(res),return_counts=True)
        inds = counts > 140
        v = dict(zip(unique[inds], counts[inds].astype(str)))
        for i,te in enumerate(v):
            with open('voc.txt','a') as w:
                w.write(str(i)+' '+te+'\n')
        ind_to_word = dict()
        with open('voc.txt') as f:
            for line in f:
                ind, word = line.strip().split()
                ind_to_word[word] = ind
    return ind_to_word

args = parser.parse_args()
bad_folders, good_folders = args.bad_folders, args.good_folders
print(bad_folders, good_folders)
x_train, x_test, y_train, y_test, df, ind_to_word = pre_process_split(bad_folders,good_folders)
df_train, df_test, ind_to_word = process_and_features(x_train, x_test, df, ind_to_word,bad_folders+good_folders)
