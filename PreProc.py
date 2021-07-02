import logging
import os
import nltk

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)


class PreProc(object):

    """
    Do some basic preprocessing from queried data
    """
    
    def __init__(self,
                 data: pd.DataFrame = pd.DataFrame()):
        
        self.data = data
        # Numeric columns from data
        self.numeric_cols = data._get_numeric_data().columns
        # Categorical Columns from data
        self.cat_cols = list(set(data) - set(self.numeric_cols))
    
    def _visualize_(self,
                    plot_data: pd.DataFrame,
                    xaxis: str,
                    yaxis: str,
                    title: str = None,
                    hue_order: list = None) -> None:
        """
        Plot (bar type) distribution of variable in the data
        Args:
            plot_data: pd.DataFrame input data
            xaxis: str column named in x axis
            yaxis: str column named in y axis
            title: str optional if want to set tiel
            hue_order: list of str to order the labels

        Returns:
        
        """
        fig, ax = plt.subplots(figsize=(20,5))
        sns.barplot(x=xaxis, 
                    y=yaxis,
                    data=plot_data,
                    hue_order=hue_order, 
                    ax=ax)
        ax.grid(True)
        ax.set(xlabel=xaxis, ylabel=yaxis)
        ax.set_title(title)
        ax.tick_params(labelrotation=90)
        
        return
        
    
    def numeric_column_dist(self,
                            column_name: str,
                            step: int = 10) -> None:
        """
        Function to bin numerical columns into group values
        and plot the count distrition
        Args:
            column_name: str, a numerical column of data
            step: int, width of a bin
        Returns:        
        """

        df = self.data[column_name].copy()   
        min_value = np.min(df)
        max_value = np.max(df)
        
        if step > max_value:
            raise("Make step smaller")        
        
        bin_range = list(range(int(min_value), int(max_value) + step, step))
        labels = [f'{bin_value}-{bin_value + step}' for bin_value in bin_range[:-1]]
        bin_value_count = pd.cut(x=df, bins=bin_range, labels=labels)
        # E.g.: Start binning with the range 0 < bin <= 10. The number with 0 values will be NaN
        # Assign those smallest bin to the smallest (first labels)
        bin_value_count = bin_value_count.replace(np.nan, labels[0])
        
        plot_data = bin_value_count.value_counts().reset_index()

        plot_data.rename(columns={'index': column_name,
                                  column_name: 'value_count'}, inplace=True)

        self._visualize_(plot_data,
                         xaxis=column_name,
                         yaxis='value_count',
                         title=f'Distribution {column_name}',
                         hue_order=labels)
        return

    
    def cat_column_dist(self,
                        column_name: str) -> None:
        """
        Function plot the count distrition of categorical variable
        If variable has too many groups (> 10). Only plot the top 10 highest counts
        Args:
            column_name: str, a numerical column of data
        Returns:        
        """       
        # No need to consider column where all values are unique 
        if self.data[column_name].nunique() == self.data.shape[0]:
            LOGGER.debug(f"all values are unique in {column_name}")
            return

        df = self.data[column_name].copy().value_counts().reset_index()
        
        df.rename(columns={'index': column_name,
                  column_name: 'value_count'}, inplace=True)
        title = None
        
        if df.shape[0] > 10:
            top_10_category = df[column_name][:10]
            df = df[df[column_name].isin(top_10_category)]
            df = df.groupby(column_name).sum().reset_index()
            title = f"Top 10 Counts of {column_name}"

        self._visualize_(df,
                         xaxis=column_name,
                         yaxis="value_count",
                         title=title)
        return

    
    def variable_exploration(self):
        """
        The function summarize all numerical/categorical columns
        to have full picture of exploration
        """
        for column in set(self.data):            
            if column in self.numeric_cols:
                self.numeric_column_dist(column)
            else:
                self.cat_column_dist(column)

    # Now below is text processing

    def string_cleaning(self,
                        string: str,
                        list_stopword: list,
                        lemmatizer: object,
                        list_kept_word: str = "[^a-zA-Z0-9 ]") -> str:
        """
        A subfunction for text_cleaning
        remove stopwords, punctuation, lower case, html, emoticons
        Args:
            string: str, input string 
            list_stopword: list of str, stop words (default English)
            lemmatizer: WordNetLemmatizer object
            list_kept_word: list of str: words to be maintained

        Returns: 
            words where unnecessary words are removed
        """
        # No need to do anything if string is empty
        if not string:
            return ''
        
        string = string
        clean_string = re.sub(list_kept_word, ' ' , string)
        clean_string = clean_string.lower()
        words = clean_string.split()
        words = [lemmatizer.lemmatize(word) for word in words if not word in list_stopword]
        clean_words = ' '.join(words)

        return clean_words

    def text_cleaning(self,
                      text_columns: list = None) -> dict:

        """
        Preprocess text to remove stopwords, punctuation,
        lower case, html, emoticons
        Args:
            text_columns: list of categorical columns from data 
        Returns:
            dictionary with:
                DataFrame: pd.DataFrame contains new column
                Cleaning Columns: List of new columns are created for this cleaning 
        """

        cat_cols = self.cat_cols
        df = self.data.copy()
        
        if text_columns is None:
            text_columns = self.cat_cols
        
        # text preprocessing
        list_stopword = stopwords.words("english")
        lemmatizer = WordNetLemmatizer()
        cleaning_columns = []
        
        for text_col in text_columns:
            
            df[f"clean_{text_col}"] = df[text_col].apply(self.string_cleaning, args = (list_stopword, lemmatizer))
            cleaning_columns.append(f"clean_{text_col}")

        return {'DataFrame': df,
                'Cleaning Columns': cleaning_columns}
    
    def word_cloud(self,
                   list_column: list,
                   max_word: int = 100) -> None:
        """
        Mining the frequency of words
        Args:
            list_column: List texted columns
            max_word: maximum number of words to be displayed in the image.
        Returns:
            dictionary with:
                DataFrame: pd.DataFrame contains new column
                Cleaning Columns: List of new columns are created for this cleaning 
        """
        for column in list_column:
            plt.subplots(figsize = (12,10))
            wordcloud = WordCloud(background_color = 'white',
                                  max_words=max_word,
                                  width=1000,
                                  height=800).generate(" ".join(self.data[column]))
            plt.imshow(wordcloud)
            plt.axis('off')
            plt.title(f'Important Words in {column}')
            plt.show()

