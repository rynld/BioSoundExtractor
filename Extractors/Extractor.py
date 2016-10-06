__author__ = 'Rey'

import numpy as np
import pandas as pd
import os
import Extractors.Spectral.mfcc as mfcc
from Extractors.RBM import RBM
import scipy.io.wavfile as wav

def get_mfcc(address):


    df = pd.DataFrame(columns=["c"+str(i) for i in range(13)])
    target = pd.Series()
    idx = 0
    for species in os.listdir(address):
        for segment in os.listdir(address + '\\' + species):
            rate,sig= wav.read(address + '\\' + species + '\\' + segment)

            res = np.mean(mfcc.mfcc(sig,rate),axis=0)

            df.loc[idx] = res
            target.loc[idx] = species
            idx += 1

    df['Category'] = target
    df.to_csv('../GeneratedFeatures/mfcc_frogs.csv')

def generate_random():
    df = pd.DataFrame(data=np.random.randn(1000,20))
    df['Category'] = np.random.randint(0,8,1000)
    df.to_csv('../GeneratedFeatures/random.csv')

def generate_rbm(database='mfcc.csv'):
    df = pd.read_csv('../GeneratedFeatures/' + database)

    X = df.drop('Category',axis=1).values
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)

    for i in range(len(std)):
        if std[i] == 0:
            std[i] = 1



    X = (X-mean)/std

    rbm = RBM(13,2048,type_visible_layer='G')
    rbm.fit(X)
    X = rbm.transform(X)

    new_df = pd.DataFrame(data=X)
    new_df['Category'] = df.Category
    new_df.to_csv('../GeneratedFeatures/rbm_mfcc.csv')


#df = pd.read_csv('../GeneratedFeatures/mfcc.csv')

generate_rbm()
#get_mfcc('C:\\Users\\Rey\\Desktop\\frogs_species')
#generate_random()
# df = pd.DataFrame(columns=["c"+str(i) for i in range(3)])
# df.to_csv('../GeneratedFeatures/mfcc.csv')

# df.loc[0] = [1,2,3]
# df.loc[1] = [1,2,3]
# serie = pd.Series()
# serie.loc[0] = 1
# serie.loc[4] = 2
# serie.loc[2] = 3
# print(serie)




