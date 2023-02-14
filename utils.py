#Utils
import numpy as np
import pandas as pd

# Preprocesing 
from sklearn.preprocessing import OneHotEncoder

#RNN WGAN 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input, Dense, Reshape
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Bidirectional

np.set_printoptions(precision = 2, suppress = True)

valid_aminoacids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P','S','T','W','Y', 'V','_'] #+ ['B','Z','J']
X = np.array(valid_aminoacids)
ohe = OneHotEncoder(sparse = False)
ohe.fit(X.reshape(-1,1))



def complete_coding(matrix,max_len):
    """

    Tranforma a minusculas todas las columnas de un Dataframe 

    Parametros
    ----------
    data: Dataframe
    Dataframe que se va a procesar

    Retorna
    -------
    df: Dataframe
    """
    agg = max_len - len(matrix)
    zeros = [0]*len(valid_aminoacids)
    if(type(matrix)!=type([])):
        matrix = list(matrix)
    for i in range(agg):
        matrix.append(zeros)
    
    return matrix

def complete_sequence(sequence,max_len):
    agregate = int(max_len - len(sequence))
    
    return sequence + "_"*agregate


def escalon(array):
  maximo = max(array)
  escalon = []
  for i in array:
    if i == maximo:
      escalon.append(1)
    else:
      escalon.append(0)

  return np.array(escalon)

def escalon_matrix(matrix):
  escalon_matrix = []
  for array in matrix:
    escalon_matrix.append(np.array(escalon(array)))
  return escalon_matrix


def get_input_generator(seq):
  #complete sequence with _
  seq = complete_sequence(seq,35)
  #coding
  coding = ohe.transform(np.array(list(seq)).reshape(-1,1))
  #flatten coding
  flatten = []
  for i in range(35):
    for j in range(21):
      flatten.append(int(coding[i][j]))
  flatten = np.array(flatten).reshape(1,-1)
  
  return flatten

  
def flatten(coding):
  flatten = []
  for i in range(35):
    for j in range(21):
      flatten.append(int(coding[i][j]))
  flatten = np.array(flatten).reshape(1,-1)
  
  return flatten


def encoding(data):
  sequences = list(complete_sequence(i,35) for i in data['sequence'])
  ohes = []
  for sequence in sequences:
      
      coding = ohe.transform(np.array(list(sequence)).reshape(-1,1))
      ohes.append(coding.reshape((35,21,1)))
    
  return np.array(ohes,dtype = "float32")