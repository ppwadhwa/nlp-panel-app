import panel as pn

from app.tokenizer import tokenizer
from app.stemmer import App_Stemmer
from app.stop_word_removal import Stopword_Remover
from app.WordEmbedding import WordEmbedder
from app.base import base_page
from app.test_train import trainer

import io
import pandas as pd
import param


class TextLoader(base_page):
    
    df = param.DataFrame()
    
    
    def __init__(self, df=None, **params):
        super().__init__(**params)
        
        self.param.name_of_page.default = 'Load Text:'
        self.load_file = pn.widgets.FileInput()
        self.load_file.link(self.df, callbacks={'value': self.load_df})
        self.header_checkbox = pn.widgets.Checkbox(name='Header included in file')
        self.continue_button.disabled = False
        
        
    @param.output('df')
    def output(self):
        return self.df
        
    
    def load_df(self, df, event):
        info = io.BytesIO(self.load_file.value)
        if self.header_checkbox.value==True:
            self.df = pd.read_csv(info)
        else:
            self.df = pd.read_csv(info, sep='\n', header = None, names=['text'])
        
        self.display_df = self.df

        
    def options_page(self):
        

        return pn.Column(
                         self.header_checkbox,
                         self.load_file
                        )
    
    
dag = pn.pipeline.Pipeline(debug=True, inherit_params=False)

dag.add_stage(
    'Load File',
    TextLoader,
    ready_parameter='ready',
    auto_advance=True
)

dag.add_stage(
    'Tokenize',
    tokenizer,
    ready_parameter='ready',
    auto_advance=True,
)

dag.add_stage(
            'Remove stop words',
            Stopword_Remover,
            ready_parameter='ready',
            auto_advance=True,
            )

dag.add_stage(
            'Stemmer',
            App_Stemmer,
            ready_parameter='ready',
            auto_advance=True,
            )

dag.add_stage(
            'Word Embedding',
            WordEmbedder,
            ready_parameter='ready',
            auto_advance=True,
            )

dag.add_stage(
            'Testing',
            trainer,
            ready_parameter='ready',
            auto_advance=True,
            )

dag.define_graph(
            {'Load File': 'Tokenize',
             'Tokenize': 'Remove stop words',
             'Remove stop words': 'Stemmer', # noqa E501
             'Stemmer': 'Word Embedding',
             'Word Embedding': 'Testing'
             }
            )


SentimentApp = pn.Column(dag.stage).servable()