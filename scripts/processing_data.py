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

#1 Create dataframe of file paths to y_values
#2 Process word in all files
#3 Split train and test sets
#4 Create the mapping of most recurrent words to overall frequencies from train set
#5 Use the mapping to map the words in the files to their index in the mapping
#6 Create dataframe from the processed files by calculating the relative frequencies of indices (words) in files
#7 Split the dataframe in train and test sets and save it

def create_y_vector(bad_folders,good_folders):
    bad_files = []
    good_files = []
    for bad_folder in bad_folders:
        bad_files += [[os.path.join(bad_folder,x),0] for x in os.listdir(bad_folder) if os.path.isfile(os.path.join(bad_folder,x))]
    for good_folder in good_folders:
        good_files += [[os.path.join(good_folder,x),1] for x in os.listdir(good_folder) if os.path.isfile(os.path.join(good_folder,x))]
    y_vec = pd.DataFrame(np.concatenate((bad_files,good_files)),columns=['file_identifier','dependent_variable'])
    return y_vec

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

def create_features_frequencies(indexed_email,ind_to_word,ind_to_freq):
    ind_to_freq = kwargs[0]
    arr_of_features = np.zeros(len(ind_to_word))
    with open(indexed_email) as f:
        content = f.read().split()
    for ind,count in np.unique(np.array(content)):
        if ind in ind_to_word:
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

def pre_process_and_mapping(y_vector,frequencies=False):
    for inp_folder in np.unique(np.array([os.path.dirname(x) for x in y_vector.file_identifier])):
        os.makedirs(os.path.join(inp_folder,'processed'))

    voc_path = 'voc.txt'
    x_train, x_test, y_train, y_test = train_test_split(y_vector['file_identifier'], y_vector.set_index('file_identifier')[['dependent_variable']],
                                                        test_size=0.25,stratify=y_vector['dependent_variable'])
    res = []
    for y in y_vector['file_identifier']:
        proc_out = process_email(y)
        with open(os.path.join(os.path.dirname(y),'processed',
                               '{}'.format(os.path.basename(y))),'w') as w:
            w.write(' '.join(proc_out))
        if y in np.append(x_train,y_train):
            res += proc_out
    print('counting terms')
    unique,counts = np.unique(np.array(res),return_counts=True)
    inds = counts > y_train.size//30
    if frequencies:
        v = dict(zip(unique[inds], (counts[inds]/train_files.shape[0]).astype(str)))
        for te,i in v:
            with open('voc_freq.txt','a') as w:
                w.write(te+' '+str(i)+'\n')
    v = dict(zip(unique[inds], np.arange(len(unique[inds])).astype(str)))
    for i,te in enumerate(v):
        with open(voc_path,'a') as w:
            w.write(te+' '+str(i)+'\n')

    return x_train, x_test, y_train, y_test

def pre_process(y_vector):
    for inp_folder in np.unique(np.array([os.path.dirname(x) for x in y_vector.file_identifier])):
        os.makedirs(os.path.join(inp_folder,'processed'))

    for y in y_vector['file_identifier']:
        proc_out = process_email(y)
        with open(os.path.join(os.path.dirname(y),'processed',
                               '{}'.format(os.path.basename(y))),'w') as w:
            w.write(' '.join(proc_out))

def load_mapping(voc_path='voc.txt'):
    ind_to_word = dict()
    with open(voc_path) as f:
        for line in f:
            word, ind = line.strip().split()
            ind_to_word[word] = ind
    return ind_to_word

def process_and_features(y_vector,folders,frequencies=False):
    print('Final processing phase')
    ind_to_word = load_mapping()
    ind_to_word_inv = {ind_to_word[x]:x for x in ind_to_word}
    if frequencies:
        ind_to_freq = load_mapping('voc_freq.txt')
        args = (ind_to_word_inv,ind_to_freq)
        trans_fun = create_features_frequencies
    else:
        args = (ind_to_word_inv,)
        trans_fun = create_features
    for inp_folder in np.unique(np.array([os.path.dirname(x) for x in y_vector.file_identifier])):
        os.makedirs(os.path.join(inp_folder,'final'))

    words = list(ind_to_word.keys())

    df_fin = pd.DataFrame(index=y_vector['file_identifier'],columns=words)
    for y in df_fin.index:
        inp_fl = os.path.join(os.path.dirname(y),'processed','{}'.format(os.path.basename(y)))
        proc_out = map_to_index(inp_fl,ind_to_word)

        with open(inp_fl.replace('processed','final'),'w') as w:
            w.write(' '.join(proc_out))
        df_fin.loc[y,:] = trans_fun(os.path.join(os.path.dirname(y),'final/{}'.format(os.path.basename(y))),*args)

    with open('final_product/data/X_and_y.pkl','wb') as wr:
        pkl.dump({'X':df_fin, 'y':y_vector.set_index('file_identifier')},wr)
    return df_fin

def save_train_test_sets(df_fin,x_train,x_test,y_train,y_test):
    df_train, df_test = df_fin.loc[x_train.values], df_fin.loc[x_test.values]
    with open('final_product/data/train_test_targets.pkl','wb') as wr:
        pkl.dump({'X_train':df_train, 'X_test':df_test,'y_train':y_train,'y_test':y_test},wr)

if __name__=='__main__':
    args = parser.parse_args()
    bad_folders, good_folders = args.bad_folders, args.good_folders
    y_vector = create_y_vector(bad_folders, good_folders)
    print(bad_folders, good_folders)
    x_train, x_test, y_train, y_test = pre_process_and_mapping(y_vector)
    df_fin = process_and_features(y_vector,bad_folders+good_folders)
    save_train_test_sets(df_fin,x_train,x_test,y_train,y_test)
