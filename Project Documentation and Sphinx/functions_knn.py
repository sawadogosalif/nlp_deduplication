from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import partial
#from functions.standardization  import *
#from functions.vectorization import *

word_blacklist = ["feat", "and", "featuring", "et", "+","&", "vs"]
regex_replace = {r"\'": "", r"\s+": " "}



def compound_similarity(row, col1: str, col2: str):
    
    """
    This function computes the a compound list of scores measuring the similarity
       between two strings  given one row of dataframe. The scores  are based on the following 6 metrics:
         - Damerau-Levenshtein - edit distance that also takes in account transpositions.
         - Jaro-Winkler - similarity based on common letters adjusted for the higher likelihood
                       spelling to be correct in the beginning of a string.
         - Jaccard - like n-grams without taking into account the cardinality (length) of the
            n-grams. Effectively, this gives n-gram similarity score for N=1.
         - Overlap - measures the 'overlap' between two strings based on the number of common
                    characters in them.
         - Ratcliff-Obershelp - takes into account the length of the fully matching substrings
        but also the number of matching characters from substrings that do not match completely.
        
    Args:
      row (DataFrame) : Dataframe to compare both dataframes
      col1 (str) : The first column (cleaned artist+ title)
      col2 (str) : The second column of strings (cleaned artist+ title)
     
    Returns:
      list(float) :  list of  similarities.  1 means that the two strings match perfectly. If Either of the two strings are
        empty, the similarity will be treated as 0.
    """
    s1 = row[col1]
    s2 = row[col2]
    
    if s1 is None:
        s1 = ""
    if s2 is None:
        s2 = ""
    if s1 == "" and s2 == "":
        return 0

    scores = [   Levenshtein.ratio(s1, s2),
                 jaro_winkler.normalized_similarity(s1, s2),
                 jaccard.normalized_similarity(s1, s2),
                 overlap.normalized_similarity(s1, s2),            
                 hamming.normalized_similarity(s1, s2),
                 fuzz.partial_ratio(s1, s2)/100
             ]
  
    return scores

def level0_knn_words(source_query, vectorizer, X_train, baseline, sample, nb_top, indices):
    
    """
    A function to run knn for a single querie after blocking_strategy.


    Args:
           

        source_query (list) : [row number , artist + title ]
        vectorizer () : type of vectorization of text 
        X_train (sparse matrix):   Document-term matrix
        baseline (dict) : subset of discogs dataset  after blocking.
        sample (DataFrame)   : a row (subset) of cdandlp 
        nb_top  (int)        : nearest neighbors
        indices (list(int)       :  list of indexes  to reduce X_train dimensions
           
    Returns:
           
        DataFrame: A table with  informations id_query', 'master_id', 'artist', 'title', 'similarity','rank' , 
                      with others text distances as hamming, levenshtein,...
    """
      
    # source_query = 'Bob Marley Is this love'
    if  len(baseline) < nb_top :
        nb_top = len(baseline)
    indice = source_query[0]
    #normalisation
    query = make_text_prep_func(source_query[1], name_column_blacklist, name_column_regex_replace)
    #vectorisation du text
    vectorized_querie = vectorizer.transform([query])
    # calcul de la similarité entre la base de ref et la base d'entrée (données tf-idf sur trigrammes)
    #on obtient un array
    mat_sim = cosine_similarity(X_train[indices], vectorized_querie)

    # fonction pour recuperer celle ayant les similarites élevées 
    #  np.argsort(x[ind]) retourne les indices du vecteur après rearranagement par ordre croissant
    def top_k(x, k):
        ind = np.argpartition(x, -1 * k)[-1 * k:]                  
        return ind[np.argsort(x[ind])]                    

    topsim = np.apply_along_axis(lambda x: top_k(x, nb_top), 0, mat_sim)
    topsim2 = pd.DataFrame( topsim, columns=['index'])

    l_sim = []
    for i in list(topsim2['index']):
        l_sim.append( float(mat_sim[i])*100.0 )
    topsim2['similarity'] = l_sim


     
    #recupération des infos d'origine
    df_nearest = pd.merge(baseline, topsim2, on = 'index', how='inner')

    df_nearest["rank"] =  df_nearest['similarity'].rank(method ='min',ascending = False).astype(int)
    df_nearest['id_query'] =sample.loc[indice,'n_ref']
    df_nearest['text_query'] = make_text_prep_func(sample.loc[indice,'text'], word_blacklist, regex_replace)
    distances = ['levenshtein', 'jaro_winkler', 'jaccard', 'overlap', 'hamming', 'fuzzy_partial']
    col_list_values = df_nearest.apply(lambda p:compound_similarity(p, "text_CLEAN", "text_query"), axis=1)
    df_nearest.drop(['index','text_query', 'text_CLEAN'], axis=1, inplace=True)

    df_nearest = pd.concat([df_nearest[['id_query', 'master_id', 'artist', 'title', 'similarity', 'rank']], 
                            pd.DataFrame(np.column_stack(col_list_values).T, columns=distances )]
                            ,axis= 1 )

    df_nearest.sort_values(by='similarity', ascending=False, inplace=True)
    new_row = pd.DataFrame(np.array([[
                                        sample.loc[indice,  'n_ref'],
                                        sample.loc[indice,  'n_ref'], 
                                        sample.loc[indice, 'artiste'],
                                        sample.loc[indice, 'titre'],
                                        float(-1),
                                        0,  1, 1, 1, 1,  1,1,]]), 
                            columns= ['id_query', 'master_id', 'artist', 'title', 'similarity', 'rank'] + distances)
    df_nearest = new_row.append(df_nearest, ignore_index=True)
    return   df_nearest
     
def level2_knn_words(start , end,  baseline,  dict_words, df_cdandlp, X_train, vectorizer) :

    """
      
    A function to run knn for a dataset of queries.
    End and start for batching  dataset.


    Args:
       
        start (int) :  1st query' index 
        end (int) :    2nd query' index
        baseline (DataFrame) : discogs' dataset . 
        dict_words (dict) : dictionary for blocking
        cdandlp (DataFrame):  cdandlp datasets

       
    Returns:
       
        save and write a list of dataframes.
      """

    df_sample= df_cdandlp[start : end]
    result = [ level1_knn_words(row_index, df_cdandlp, baseline, dict_words, X_train, vectorizer)  for row_index in range(start, end) ]
      
    with open(REP_INTERMED + str(end) + '_All_words_knn.pkl', 'wb') as  f1:
        pickle.dump(result, f1) 

def level1_knn_words(index, df, baseline, dict_words,  X_train,vectorizer):
    """
    A function to run the method level0_knn_words for a single querie by with blocking strategy.


    Args:
       

        index (int) : row identification 
        df_baseline (DataFrame) : discogs data 
        df (pandas.DataFrame):  cdandlp datasets
        dict_words (dict) : dictionary for blocking 
       
    Returns:
       
        DataFrame: A table with  informations id_query', 'master_id', 'artist', 'title', 'similarity','rank' , 
        with others text distances as hamming, levenshtein,...
    """

    row = df.loc[index, 'text']
    # retrieve the list of indices by word
    indices_list = [dict_words.get(key) for key in row.split()]
    distances = ['levenshtein', 'jaro_winkler', 'jaccard', 'overlap', 'hamming', 'fuzzy_partial']
    while None in indices_list:
        indices_list.remove(None)
    # convert the list of lists to flat list
    indices_list_flat = list(set([item for sublist in indices_list for item in sublist]))
    if indices_list_flat == [] :
        new_row = pd.DataFrame(np.array([[
                                        df.loc[index,  'n_ref'],
                                        df.loc[index,  'n_ref'], 
                                        df.loc[index, 'artiste'],
                                        df.loc[index, 'titre'],float(-1), 0,   1,   1, 1,  1,  1,    1,]]),
                               columns= ['id_query', 'master_id', 'artist', 'title', 'similarity','rank'] + distances    

                            )
        return new_row
    # else
    # match indexes with baseline indexes
    df_baseline = baseline.iloc[indices_list_flat,].reset_index(drop=True)
    df_sample = df.iloc[[index,]].reset_index(drop='True')
    df_baseline['index'] = df_baseline.index
    answer = level0_knn_words([0, df_sample.loc[0,'text']] ,indices =indices_list_flat ,vectorizer=vectorizer, X_train=X_train, baseline=df_baseline.copy(), sample =df_sample.copy(), nb_top=50)
      
    return answer
