import os  
REP_PROJET = os.getcwd()
REP_INTERMED  = REP_PROJET + '/Intermed/'

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np
import re



def text_to_vect(source_query:str, vectorizer):

    """
    A function to Standarize a string and vectorized it.
    
    Args:
        source_query (str) : concatenation of Artist and title.
        vectorizer (vect)  : vectorizer of discogs dataset
        
    Returns:
         vector : document-term matrix.

    """

    #normalisation
    query = make_text_prep_func(source_query, name_column_blacklist, name_column_regex_replace)

    #vectorisation du text
    vectorized_querie = vectorizer.transform([query])
   
    return  vectorized_querie
    

def create_vectorizer_and_vectorized_db(df_items, analyzer, ngram_range):

    """
    This function returns the vectorized reference base + the vectorize itself.
    
    Args:
        df_items(DataFrame) : Baseline Discogs.
        analyzer (str)  : {'word', 'char', 'char_wb'} 
        ngram_range (tuple) : Range of n-values for different n-grams to be extracted
    
    Returns:
            (matrix, vectorizer) :  Save/write to pickle the method of vectorization and the X_train

    """       
    

    #creation du vectorizer et de la base vectorisÃ©e
    print("**** creation du vectorizer")
    if analyzer == 'word' :
      vectorizer = TfidfVectorizer(analyzer = 'word', use_idf = False)
      X_train = vectorizer.fit_transform([ document for document in df_items['text_CLEAN'] ])
      
    else :
      vectorizer = TfidfVectorizer(analyzer = 'char', ngram_range=ngram_range , use_idf = False)
      X_train = vectorizer.fit_transform(df_items['text_CLEAN'].str.replace(' ', ''))
    
    #on "exÃ©cute" le vectorizer et en sortie on a une matrice sparse + le vocabulaire associÃ©
    print("nb features:", len(vectorizer.get_feature_names()))

    print("**** sauvegarde des fichiers en pickle pour la prochaine fois")
    suffix_gram = re.sub(r"[(,)]", "", str(ngram_range)).replace(' ','_')
    with open(REP_INTERMED + analyzer + '_' + suffix_gram+ '_Xtrain.pkl', 'wb') as  f1 :
      pickle.dump(X_train, f1) 

    with open(REP_INTERMED + analyzer + '_'+  suffix_gram+ '_vectorizer.pkl', 'wb') as f2 :
      pickle.dump(vectorizer, f2) 
      

def reload_data_and_vectorizer(analyzer = None, ngram_range=None) :

  """
  This function to import the pickles (vectorizer and the matrix Sparse of the documents).
    
  Args:
        df_items(DataFrame) : Baseline Discogs.
        analyzer (str)  : {'word', 'char', 'char_wb'} 
        ngram_range (tupe) : Rrange of n-values for different n-grams to be extracted
    
  Returns:
          pickle: (matrix, vectorizer)

  """     


  if analyzer== None and ngram_range==None :
    
     with open(REP_INTERMED + 'Xtrain.pkl', 'rb') as f:
         X_train = pickle.load(f)
    
     with open(REP_INTERMED + 'vectorizer.pkl', 'rb') as handle:
         vectorizer = pickle.load(handle)
  else :

      suffix_gram = re.sub(r"[(,)]", "", str(ngram_range)).replace(' ','_')

      with open(REP_INTERMED + analyzer + '_' + suffix_gram+ '_Xtrain.pkl', 'rb') as f:
         X_train = pickle.load(f)
    
      with open(REP_INTERMED + analyzer + '_' + suffix_gram+ '_vectorizer.pkl', 'rb') as handle:
         vectorizer = pickle.load(handle)        
        
  return  X_train, vectorizer


