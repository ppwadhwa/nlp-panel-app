import panel as pn
from nltk.stem import (PorterStemmer, SnowballStemmer)
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer


from app.test_train import trainer

import io
import pandas as pd
import param


class PreProcessor(param.Parameterized):
    
    df = param.DataFrame()
    name_of_page = param.String(default = 'Name of page')
    display_df = param.DataFrame(default = pd.DataFrame())
    stopword_df = param.DataFrame(default = pd.DataFrame())
    
    stopwords = param.List(default = [])
    X = param.Array(default = None)
    
    ready = param.Boolean(
        default=False,
        doc='trigger for moving to the next page',
        )   
    
    def __init__(self, df=None, **params):
        super().__init__(**params)
        
        self.search_pattern_input = pn.widgets.TextInput(name='Search Pattern', value = '\w+', width = 100)
        
        self.stem_choice = pn.widgets.Select(name='Select', options=['Porter', 'Snowball'])
        self.load_file = pn.widgets.FileInput()
        self.load_file.link(self.df, callbacks={'value': self.load_df})
        self.header_checkbox = pn.widgets.Checkbox(name='Header included in file')
        
        self.load_words_button = pn.widgets.FileInput()
        self.load_words_button.link(self.stopwords, callbacks={'value': self.load_stopwords})
        
        self.continue_button = pn.widgets.Button(name='Continue', 
#                                                  disabled = True,
                                                 width = 100,
                                                 button_type='primary')
    
        self.continue_button.on_click(self.continue_ready)
    
    def continue_ready(self, event):
        #need to do all the preprocessing to go onto train-test
        tokenizer = RegexpTokenizer(self.search_pattern_input.value)
        if self.stem_choice.value == 'Porter':
            stemmer = PorterStemmer() 
        else:
            stemmer = SnowballStemmer()
            
        if self.we_model.value == 'SKLearn Count Vectorizer':
            # Create a vectorizer instance
            vectorizer = CountVectorizer(max_features=1000)
            
        corpus = []

        #loop through each line of data
        print('check1')
        for n in range(len(self.display_df)):  
            print('check1b')
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
            
        self.ready=True
        
        
    @param.output('X', 'display_df')
    def output(self):
        return self.X, self.display_df
        
    def load_text_page(self):
        
        return pn.Row(
                pn.Column(
                    pn.pane.Markdown(f'##Load Text:'),
                    pn.Column(
                         self.header_checkbox,
                         self.load_file
                        ),
                ),
                pn.Column(
                    pn.Spacer(height=52),
                    self.df_pane,
#                     self.continue_button_pane,
                    
                )
        
        )
    
    def tokenize_option_page(self):
        
        return pn.Column(
                    pn.pane.Markdown(f'##Tokenize options:'),
                    pn.WidgetBox(self.search_pattern_input,
                                    height = 300,
                                    width = 300
        
                                )
                )
    
    
    def stemmer_page(self):
        
        return pn.Column(
                    pn.pane.Markdown(f'##Stemmer options:'),
                    pn.WidgetBox(self.stem_choice,
                height = 300,
                width = 300)
                )
    @param.depends('stopword_df')
    def remove_stopwords_page(self):
        
        
        return pn.Row(
                pn.Column(
                    pn.pane.Markdown(f'##Load Stopwords:'),
                    pn.WidgetBox(self.load_words_button,
                                    height = 300,
                                    width = 300
        
                    )
                ),
                pn.Column(
                    pn.Spacer(height=52),
                    pn.WidgetBox(self.stopword_df,
                           height = 300,
                           width = 400)
#                     self.continue_button_pane,
                    
                )
        
        )
           
     
    def word_embedding_page(self):
        
        self.we_model = pn.widgets.Select(name='Select', options=['SKLearn Count Vectorizer'])
        
        return pn.Column(
                    pn.pane.Markdown(f'##Choose embedding model:'),
                    pn.WidgetBox(self.we_model,
#                             self.we_button,
                            height = 300,
                            width = 300
        
                    )
        
                )
    
        
    def load_stopwords(self, stopwords, event):
        info = io.BytesIO(self.load_words_button.value)
        self.stopwords = pd.read_pickle(info)
        self.stopword_df = pd.DataFrame({'stop words': self.stopwords})
    
    def load_df(self, df, event):
        info = io.BytesIO(self.load_file.value)
        if self.header_checkbox.value==True:
            self.df = pd.read_csv(info)
        else:
            self.df = pd.read_csv(info, sep='\n', header = None, names=['text'])
        
        self.display_df = self.df
        
        
    @param.depends('display_df')
    def df_pane(self):
        return pn.WidgetBox(self.display_df,
                           height = 300,
                           width = 400)
    
    
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
        

    
    
dag = pn.pipeline.Pipeline(debug=True, inherit_params=False)

dag.add_stage(
    'Preprocess',
    PreProcessor,
    ready_parameter='ready',
    auto_advance=True
)

dag.add_stage(
            'Testing',
            trainer,
            ready_parameter='ready',
            auto_advance=True,
            )

dag.define_graph(
            {'Preprocess': 'Testing',
             }
            )


SentimentApp = pn.Column(dag.stage).servable()