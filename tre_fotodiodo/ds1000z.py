import numpy as np
import pandas as pd

"""
Questo modulo contiene delle funzioni per analizzare i dati presi da un oscilloscopio DS1000Z
nella terza epserienza del corso di lab 3 di fisica Unipg 2k24-25.

Autore: Matteo Cecchini
"""


def csv_formatter(df: pd.DataFrame):
    '''
    Questa funzione formatta un csv di dati in maniera più comprensibile secondo me (l'autore).
    
    Return: un dataframe con colonne il tempo di ogni acquisizione e il corrispondente voltaggio
            del primo canale (dal momento che uso solo io il modulo quando avrò bisogno di generalizzare lo farò).
    '''
    cols = ("time", "ch1_volt")
    increment = df["Increment"][0]
    df.drop(index=0, inplace=True)
    df.drop(columns=df.columns[-2:], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.columns = cols
    for i in cols:
        df[i] = pd.to_numeric(df[i]) 
    df["time"] *= increment
    return df


def split_and_mean(df, col, colsave):
    '''
    Questa funzione genera i punti centrali di ogni minimo di un set di dati di 12 secondi di oscillazione
    del pendolo. Prima divide l'array dei tempi prendendo tutti i valori sotto la media di voltaggio, poi taglia
    guardando alle differenze temporali di istanti successivi, poi riporta un array con media e deviazione standard
    di ogni minimo.
    '''
    mn = np.mean(df[col])
    a = df[df[col] < mn][colsave]
    b = a.index
    mn, sd = [], []
    j = 0
    for i in range(1, len(b)):
        if b[i] != 1 + b[i - 1]:
            mn.append( np.mean( a[ b[j:i] ] ) )
            sd.append( np.std( a[ b[j:i] ] ) )
            j = i
    mn.append( np.mean( a[ b[j:] ] ) )
    sd.append( np.std( a[ b[j:] ] ) )
    return np.array([mn, sd])
    
def time_deltas(x):
    '''
    Questa funzione dà un array con le differenze temporali di minimi successivi,
    dato un array di minimi con relative std.
    '''
    dt = ( x[0][1:] - x[0][:-1] ) * 2
    dtstd = np.sqrt( x[1][1:]**2 + x[1][:-1]**2 ) * 2
    return np.array([dt, dtstd])

def drop_g(t, l):
    '''
    Funzione per calcolare g e l'errore associato dato l'array con le stime dei periodi e 
    le loro incertezze, e data lastima della lunghezza del pendolo con la sua incertezza.
    '''
    tm = np.mean(t[0])
    tmstd = np.sqrt( np.sum(t[1]) ) / len(t[1])
    
    lm = np.mean(l)
    lmstd = np.std(l, ddof=1)
    
    g = lm*(2*np.pi / tm)**2
    gstd = np.sqrt( ( lmstd*(2*np.pi/tm)**2 )**2 + ( tmstd*( (lm*(2*np.pi)**2) / tm**3 ) )**2 )
    return g, gstd