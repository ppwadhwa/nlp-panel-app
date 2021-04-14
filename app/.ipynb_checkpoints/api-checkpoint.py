import panel as pn
from nltk.stem import (PorterStemmer, SnowballStemmer)
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer


from app.test_train import trainer

import io
import pandas as pd
import param


class PreProcessor(param.Parameterized):
    
    # df will be the variable holding the dataframe of text
    df = param.DataFrame()
    # title to display for each tab
    name_of_page = param.String(default = 'Name of page')
    # dataframe to display.
    display_df = param.DataFrame(default = pd.DataFrame())
    # stopword_df is the dataframe containing the stopewords
    stopword_df = param.DataFrame(default = pd.DataFrame())
    
    stopwords = param.List(default = [])
    X = param.Array(default = None)
    
    ready = param.Boolean(
        default=False,
        doc='trigger for moving to the next page',
        )   
    
    def __init__(self, **params):
        super().__init__(**params)
        
        
        
        # button for the pre-processing page
        self.continue_button = pn.widgets.Button(name='Continue',
                                                 width = 100,
                                                 button_type='primary')

        self.continue_button.on_click(self.continue_ready)
        
        # load text widgets 
        self.header_checkbox = pn.widgets.Checkbox(name='Header included in file')
        self.load_file = pn.widgets.FileInput()
        self.load_file.link(self.df, callbacks={'value': self.load_df})
        self.header_checkbox = pn.widgets.Checkbox(name='Header included in file')
        
        # tokenize widgets
        self.search_pattern_input = pn.widgets.TextInput(name='Search Pattern', value = '\w+', width = 100)
        
        # remove stop words widgets
        self.load_words_button = pn.widgets.FileInput()
        self.load_words_button.link(self.stopwords, callbacks={'value': self.load_stopwords})
        
        # stem widgets
        self.stem_choice = pn.widgets.Select(name='Select', options=['Porter', 'Snowball'])
        
        # embedding widgets
        
        self.we_model = pn.widgets.Select(name='Select', options=['SKLearn Count Vectorizer'])

        
    @param.output('X', 'display_df')
    def output(self):
        return self.X, self.display_df
    
    
    @param.depends('display_df')
    def df_pane(self):
        return pn.WidgetBox(self.display_df,
                           height = 300,
                           width = 400)
    
    # load text page functions
    #-----------------------------------------------------------------------------------------------------
    def load_df(self, df, event):
        info = io.BytesIO(self.load_file.value)
        if self.header_checkbox.value==True:
            self.df = pd.read_csv(info)
        else:
            self.df = pd.read_csv(info, sep='\n', header = None, names=['text'])
        
        self.display_df = self.df
    
    def load_text_page(self):
        helper_text = (
            "This simple Sentiment Analysis NLP app will allow you to select a few different options " +
            "for some preprocessing steps to prepare your text for testing and training. " +
            "It will then allow you to choose a model to train, the percentage of data to " +
            "preserve for test, while the rest will be used to train the model.  Finally, " +
            "some initial metrics will be displayed to determine how well the model did to predict " +
            "the testing results." +
            " " +
            "Please choose a csv file that contains lines of text to analyze.  This text should " +
            "have a text column as well as a sentiment column.  If there is a header included in the file, " +
            "make sure to check the header checkbox."
        )
        return pn.Row(
                pn.Column(
                    pn.pane.Markdown(f'##Load Text:'),
                    pn.Column(
                        helper_text,
                         self.header_checkbox,
                         self.load_file
                        ),
                ),
                pn.Column(
                    pn.Spacer(height=52),
                    self.df_pane,
                    
                )
        
        )

    #-----------------------------------------------------------------------------------------------------
    
    # tokenize page options
    #-----------------------------------------------------------------------------------------------------
    def tokenize_option_page(self):
        
        help_text = ("Tokenization will break your text into a list of single articles " +
            "(ex. ['A', 'cat', 'walked', 'into', 'the', 'house', '.']).  Specify a regular " +
            "expression (regex) search pattern to use for splitting the text.")
        
        return pn.Column(
                    pn.pane.Markdown(f'##Tokenize options:'),
                    pn.WidgetBox(help_text, self.search_pattern_input,
                                    height = 300,
                                    width = 300
        
                                )
                )
    
    #-----------------------------------------------------------------------------------------------------
    
    
    # remove stopwords page 
    #-----------------------------------------------------------------------------------------------------
    
    def remove_stopwords_page(self):
        
        help_text = (
            "Stop words are words that do not add any value to the sentiment of the text. " +
            "Removing them may improve your sentiment results.  You may load a list of stop words " +
            "to exclude from your text."
        )
        return pn.Row(
                pn.Column(
                    pn.pane.Markdown(f'##Load Stopwords:'),
                    pn.WidgetBox(help_text, self.load_words_button,
                                    height = 300,
                                    width = 300
        
                    )
                ),
                pn.Column(
                    pn.Spacer(height=52),
                    pn.WidgetBox(self.stopword_df,
                           height = 300,
                           width = 400)
                    
                )
        )
    
    def load_stopwords(self, stopwords, event):
        info = io.BytesIO(self.load_words_button.value)
        self.stopwords = pd.read_pickle(info)
        self.stopword_df = pd.DataFrame({'stop words': self.stopwords})

    #-----------------------------------------------------------------------------------------------------
    
    # stemming page 
    #-----------------------------------------------------------------------------------------------------
    
    def stemmer_page(self):
        help_text = (
            "Stemming is a normalization step for the words in your text.  Something that is " +
            "plural should probably still be clumped together with a singular version of a word, " +
            "for example.  Stemming will basically remove the ends of words.  Here you can choose " + 
            "between a Porter Stemmer or Snowball Stemmer. Porter is a little less aggressive than " +
            "Snowball, however, Snowball is considered a slight improvement over Porter."
        )
        return pn.Column(
                    pn.pane.Markdown(f'##Stemmer options:'),
                    pn.WidgetBox(help_text, self.stem_choice,
                height = 300,
                width = 300)
                )
    
    #-----------------------------------------------------------------------------------------------------
    
    # embedding page 
    #-----------------------------------------------------------------------------------------------------
    
    def word_embedding_page(self):
        
        help_text = ("Embedding the process of turning words into numerical vectors. " +
                    "There have been several algorithms developed to do this, however, currently in this " +
                    "app, the sklearn count vectorizer is available. This algorithm will return a sparse " +
                    "matrix represention of all the words in your text."
                    )
        
        
        
        return pn.Column(
                    pn.pane.Markdown(f'##Choose embedding model:'),
                    pn.WidgetBox(help_text, self.we_model,
                            height = 300,
                            width = 300
        
                    )
        
                )
    
    #-----------------------------------------------------------------------------------------------------
          
    def continue_ready(self, event):

        # Set up for tokenization
        tokenizer = RegexpTokenizer(self.search_pattern_input.value)

        # Set up for stemming
        if self.stem_choice.value == 'Porter':
            stemmer = PorterStemmer() 
        else:
            stemmer = SnowballStemmer()

        # Set up for embedding
        if self.we_model.value == 'SKLearn Count Vectorizer':
            # Create a vectorizer instance
            vectorizer = CountVectorizer(max_features=1000)

        corpus = []
        #loop through each line of data
        for n in range(len(self.display_df)):  
            sentence = self.display_df.iloc[n].text

            #1. Tokenize
            tokens = tokenizer.tokenize(sentence)

            #2. remove stop words
            tokens_no_sw = [word for word in tokens if not word in self.stopwords]

            #3. stem the remaining words
            stem_words = [stemmer.stem(x) for x in tokens_no_sw]

            #Join the words back together as one string and append this string to your corpus.
            corpus.append(' '.join(stem_words))

        X = vectorizer.fit_transform(corpus).toarray()
        labels = self.display_df['sentiment']

        xlist = []
        for n in range(len(X)):
            xlist.append(list(X[n]))
        self.X = X
        self.display_df = pd.DataFrame({'embeddings': xlist, 'sentiment': labels})
        
        self.ready = True
    
    def panel(self):
        
        return pn.Column(
            pn.Tabs(
                ('Load Text', self.load_text_page),
                ('Tokenize', self.tokenize_option_page),
                ('Remove Stopwords', self.remove_stopwords_page),
                ('Stem', self.stemmer_page),
                ('Embed', self.word_embedding_page)
                ),
            self.continue_button
        )
        
