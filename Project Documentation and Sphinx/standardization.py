import numpy as np
import re
import pickle
from unidecode import unidecode
import string
# Compact Language Detector v3 is a very fast and performant algorithm by Google for language detection: more info here: https://pypi.org/project/pycld3/
#import cld3
#import Levenshtein

from textdistance import damerau_levenshtein, jaro_winkler, sorensen_dice, jaccard, overlap, ratcliff_obershelp,hamming
from fuzzywuzzy import fuzz
# Snowball stemmer was chosen in favor of Porter Stemmer which is a bit more aggressive and tends to remove too much from a word
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download("punkt")
nltk.download("stopwords")

name_column_blacklist = ["feat", "and", "featuring", "et", "+","&", "vs"]
name_column_regex_replace = {r"\'": "", r"\s+": " "}

STEMMER_EN = SnowballStemmer(language='english')
STEMMER_FR = SnowballStemmer(language='french')


def make_text_prep_func(row, 
                        word_blacklist,
                        regex_replace, 
                        colonne:str =None) :
    """
    This function treats the input string by going through the following steps:
        - Language detection
        - Remove punctuation and special characters
        - Tekenization
        - Stop-word removal
        - Stemming
        - ASCII folding.
    
    Args:
      
        row (str) : The input string to be treated.
        word_blacklist (list[str]) : additional stop-word
        regex_replace (Dict[str, str]) ::characters to remove
         colonne :  name of the colonne to prepare.  Default = None,If None, the input is a string.
      
    Returns:
      str : The treated version of the string. 
    """
    

    if colonne == None :
      s = str(row)
    else :
      s=row[colonne]
    
    # in the default case use the English stop-words and stemmer
    stemmer = STEMMER_EN
    stop_words =  word_blacklist

    
    # convert to lowercase, just to be sure :)
    s = s.lower()
    
    # check if the language is French and switch to the French
    s_lang = cld3.get_language(s)
    if s_lang[0]=="fr":
      stemmer = STEMMER_FR
      stop_words = word_blacklist


    # remove punctuation
    s_clean = s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # tokenize the string into words
    s_tokens = word_tokenize(s_clean)

    # remove the stop-word tokens

    s_tokens_no_stop = [word for word in s_tokens if word not in stop_words]
    
    # join the stemmed tokens together and ASCII fold
    s_tokens_stemmed = [stemmer.stem(word) for word in s_tokens_no_stop]
    s_ascii = unidecode(" ".join(s_tokens_stemmed))
    
    for regex, replace in regex_replace.items():
      s_ascii = re.sub(regex, replace, s_ascii)

    return(s_ascii.strip())
    
    
    
def recode_pressage(df):
    
    """ This standarize the pressage for cdandlp dataset
    
    Args:
        df (DataFrame) : cdandlp datsets for matching
        
    Returns:
        DataFrame : add a column 'pays_clean'
    """
    map = {
          'Suisse' : 'Switzerland',
         'Swiss' : 'Switzerland',
         'Union Europeenne' : 'Europe', 
         'SB' : 'Solomon Islands',
         'Fl' : 'Liechtenstein',
         'Ukr' : 'Ukraine',
         'Ukraineaineaineaineaineaine' : 'Ukraine',
         'Ussrsia' : 'Ussr',
         'French' : 'France',
         'Francais' : 'France',
         'Français' : 'France',
         'Fr -' : 'France',
         'Russia' :'Ussr',
         'United Kingdom' : 'Uk',
         'United Kingdom' : 'Uk',
         'Angleterre' : 'Uk',
         'Royaume Uni' : 'Uk',
         'Italie': 'Italy',
         'Italia': 'Italy',
         'Anglais' : 'Uk',
         'England': 'Uk',
         'Usa': 'Us',
         'U.S.A' : 'Us',
         'Us.'   : 'Us',
         'Belgique' :'Belgium',
         'Belge'  :'Belgium',
         'España': 'Spain',
         'Espagne' : 'Spain',
         'Europerope' : 'Europe',
         'Grèce' : 'Greece',
         'U.K.' : 'Uk',
         'Holland' : 'Netherlands',
         'Netherlandse' : 'Netherlands',
         'Deutschland' : 'Germany',
         'Germany.' :	 'Germany',
          '. Germany' :	 'Germany',
          'Japon' : 'Japan',
          'E.U' : 'Europe',
          'Al'  : 'Albania',
          'Eu' : 'Europe',
          'London' : 'Uk',
           'Gb'  : 'Uk',
           'é'  : 'e',
           'Ru' : 'Ussr',
           'Nederland' : 'Netherlands',
           'Netherlandse' : 'Netherlands',
           'Brasil' : 'Brazil',
           'Bresil' :'Brazil',
           'Coreen' : 'South Korea',
           'Australiia' : 'Australia',
           'Australie' : 'Australia',
           'Pays Bas' : 'Netherlands',
           'Etats Unis' : 'Us',
           'United States' : 'Us',
           'Esp' :'Spain',
           'G.B' :'Uk',
           'Swe' :  'Sweden', 
           'Original'  : '',
           '(Original)' : ''
    }


    ## standarized  countries' Format 
    df = df.replace({'pressage' : map},regex=True)
    # ++++++++++++++++++ many digits in column pressage ++++++++++++++++++++++++++++
    df['pressage'] = df['pressage'].str.replace('\d+', '')
    # ++++++++++++++++++ Needless words ++++++++++++++++++++++++++++
    df['pressage'] = df['pressage'].str.replace('(Original)', '')
    df['pressage'] = df['pressage'].str.replace('Made In', '')
    df['pressage'] = df['pressage'].str.replace('Original', '')
    df['pressage'] = df['pressage'].str.replace('Limited', '')
    df['pressage'] = df['pressage'].str.replace('Press', '')
    df['pressage'] = df['pressage'].str.replace('Biem', '')
    df['pressage'] = df['pressage'].str.replace('Biem', '')
    # ++++++++++++++++++ Brute recoding++++++++++++++++++++++++++++
    df['pressage'] = df['pressage'].str.replace('Europerope', 'Europe')
    df['pressage'] = df['pressage'].str.replace('Netherlandse', 'Netherlands')
    df['pressage'] = df['pressage'].str.replace('Nl', 'Netherlands')
    df['pressage'] = df['pressage'].str.replace('Dutch', 'Netherlands')
    df['pressage'] = df['pressage'].str.replace('Swedenden', 'Sweden')
    df['pressage'] = df['pressage'].str.replace('Albanialemagn', 'Albania')
    df['pressage'] = df['pressage'].str.replace('Albanial', 'Albania')
    df['pressage'] = df['pressage'].str.replace('Albaniae', 'Albania')
    df['pressage'] = df['pressage'].str.replace('WesGermany', 'Germany')
    # ++++++++++++++++++++ NEXT STEP ++++++++++++++++++++++++++++++++++++
    df['pressage'] = df['pressage'].str.strip()
    df['pressage'] = df['pressage'].str.replace('Usr', 'Ussr')
    df['pressage'] = np.where( df['pressage'] == 'German' , 'Germany',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Gdr' , 'Germany',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Bel' , 'Belgium',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'European Union' , 'Europe',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Swedenden' , 'Sweden',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Italyn' , 'Italy',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'U.K' , 'Uk',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Netherlandsais' , 'Netherlands',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Netherlandsland' , 'Netherlands',  df['pressage'])   
    df['pressage'] = np.where( df['pressage'] == 'Albaniamand', 'Albania',df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Franc -', 'France -',df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Stereo Germany'	, 'Germany',df['pressage'])
    df['pressage'] = np.where( df['pressage'] =='U.S', 'Us', df['pressage'])
    df['pressage'] = np.where( df['pressage'] =='Fra', 'France', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Deutchland - Deutchland', ' Germany', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Autriche', 'Austria', df['pressage'])
    df['pressage'] = np.where( df['pressage'] =='Made In Us', 'Us', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Canadien', 'Canada', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Can', 'Canada', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Pl', 'Poland', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == '- Ue', 'Europe', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Epc', 'Europe', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Ita' , 'Italy', df['pressage'])
    # ++++++++++++++++++++ NEXT STEP ++++++++++++++++++++++++++++++++++++
    df['pressage'] = df['pressage'].str.strip('.()- ')
    df['pressage'] = np.where( df['pressage'] == 'U.S' ,'Us', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Hol' , 'Netherlands',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Ho' , 'Netherlands',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Ukraineaine' , 'Ukraine',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Atl Germany' , 'Germany',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Fr' , 'France',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] ==  'Ecc', 'Europe', df['pressage'])
    df['pressage'] = np.where( df['pressage'] ==  'Eec', 'Europe', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Albaniagerie' , 'Albania',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Ue', 'Europe', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'U.E', 'Europe', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Ca', 'Canada', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Europeropa', 'Europe', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'The Netherlands', 'Netherlands', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Holl', 'Netherlands', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Netherland', 'Netherlands', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'France C', 'France', df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'German' , 'Germany',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'De' , 'Germany',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Ger' , 'Germany',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'Atl Germany' , 'Germany',  df['pressage'])
    df['pressage'] = np.where( df['pressage'] == 'France Sans' , 'France',  df['pressage'])
    df['pressage'] = np.where( df['pressage']	== 'Türkiye', 'Turkey', df['pressage'])
    df['pays']  = df['pressage'].apply(lambda p : list(p.split('-')))
        
    with open(REP_INTERMED + 'list_country:.pkl', 'rb') as f:
        list_country = pickle.load(f)          

    def recode_country(elements):

        for element in elements :
            if element.strip() in list_country:
                return  element
        return 'NaN'

    df['pays_clean'] = df['pays'].apply(recode_country)
        
    return df