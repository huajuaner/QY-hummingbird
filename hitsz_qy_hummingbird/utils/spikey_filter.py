

'''
This filter assumes that the signal follows a harmonic signal
and removes all the points outside
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class SpikeyFilter():

    def __init__(self,
                 DataToBeProcessed,
                 SPAN):
        self.DELTA = 0
        df = pd.DataFrame(DataToBeProcessed, columns=['Original'])
        df['Clipped'] = self.clip_data(df['Original'])
        df['EWMA'] = self.ewma_fb(df['Clipped'], span=SPAN)
        df['Removed'] = self.remove_outliers(df['Original'], df['EWMA'], delta=self.DELTA)
        df['Interpolated'] = df['Removed'].interpolate()
        self.df = df

    def theProcessedData(self):
        return self.df['Interpolated'].values

    def clip_data(self,
                  unclippedData):
        unclippedData = np.array(unclippedData)
        theDiffedData = np.concatenate(([0], np.diff(unclippedData)))

        theTop5perRemoved = self.remove_top_k(data=theDiffedData,
                                              k=int(len(theDiffedData) / 40))

        theDistance = np.var(theTop5perRemoved)
        theMeanDiff = np.sum(theTop5perRemoved) / len(theTop5perRemoved)

        a = np.sqrt(theDistance * 2)
        HIGH_CLIP = theMeanDiff + 10 * a
        LOW_CLIP = theMeanDiff - 10 * a
        self.DELTA = HIGH_CLIP

        cond_high_clip = (theDiffedData > HIGH_CLIP) | (theDiffedData < LOW_CLIP)
        np_clipped = np.where(cond_high_clip, np.nan, unclippedData)
        return np_clipped.tolist()

    def ewma_fb(self,
                df_column,
                span):
        fwd = pd.Series.ewm(df_column, span=span).mean()
        bwd = pd.Series.ewm(df_column[::-1], span=10).mean()
        stacked_ewma = np.vstack((fwd, bwd[::-1]))
        fb_ewma = np.mean(stacked_ewma, axis=0)
        return fb_ewma.flatten().tolist()

    def remove_outliers(self,
                        spikey,
                        fbewma,
                        delta):
        np_spikey = np.array(spikey)
        np_fbewma = np.array(fbewma)
        cond_delta = (np.abs(np_spikey - np_fbewma) > delta)
        np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
        return np_remove_outliers

    def remove_top_k(self,
                     data,
                     k):
        kth_largest = np.partition(data, -k)[-k]
        kth_smallest = np.partition(data, k)[k]
        mask = (data <= kth_largest) & (data > kth_smallest)
        return data[mask]
