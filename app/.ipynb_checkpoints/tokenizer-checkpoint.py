import panel as pn
import param
import pandas as pd

import sparknlp
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer

from app.base import base_page
from app.base import get_all_lines
import pandas as pd
from pyspark.sql.dataframe import DataFrame as sdf




class tokenizer(base_page):
    
    search_pattern_input = pn.widgets.TextInput(name='Search Pattern', value = '\w+', width = 100)
    spark_df = param.ClassSelector(
        class_= sdf
    )
    df = param.DataFrame()
    
    def __init__(self, **params):
        super().__init__(**params)
        print('initializing')
        self.spark = sparknlp.start()
        print('started spark')
        self.param.name_of_page.default = 'Tokenizer'

        self.tokenize_button = pn.widgets.Button(name='Tokenize', button_type='primary')
        self.tokenize_button.on_click(self.tokenize)
        
        self.display_df = self.df
        
        
        
    def options_page(self):
        
        return pn.WidgetBox(self.search_pattern_input,
                            self.tokenize_button,
                height = 300,
                width = 300
        
        )
    
    
    def tokenize(self, event):
        print('entered tokenizer')
        
        documentAssembler = DocumentAssembler()

        documentAssembler.setInputCol('text')

        documentAssembler.setOutputCol('document')
        
        self.spark_df = self.spark.createDataFrame(self.df.astype(str))
        self.spark_df=documentAssembler.transform(self.spark_df)
        tokenizer = Tokenizer()
        tokenizer.setInputCols(['document'])
        tokenizer.setOutputCol('token')
        tokenizer.setTargetPattern(self.search_pattern_input.value)
        token_df=tokenizer.fit(self.spark_df)
        current_df = token_df.transform(self.spark_df) 
        self.spark_df = current_df

        

        self.display_df = get_all_lines(self.spark_df, 'token.result', col = 'token')
        self.continue_button.disabled = False

        
    @param.output('spark_df', 'display_df')
    def output(self):
        return self.spark_df, self.display_df