import pandas as pd
from sparknlp.annotator import StopWordsCleaner
from app.base import base_page
import panel as pn
import param
from app.base import get_all_lines

from pyspark.sql.dataframe import DataFrame as sdf


class Stopword_Remover(base_page):
    
    spark_df = param.ClassSelector(
        class_= sdf
    )
    
    display_df = param.DataFrame(default = pd.DataFrame())
    
    
    def __init__(self, **params):
        super().__init__(**params)
        
        self.stopwords = pd.read_pickle('stopwords.txt')

        self.param.name_of_page.default = 'Remove Stop Words'

        #Note: once stopwords are loaded, display them in the dataframe display
        self.load_words_button = pn.widgets.Button(name='To DO: Load Stop Words', 
                                                   disabled = True,
                                                   button_type='primary')
        
        self.remove_button = pn.widgets.Button(name='Remove', button_type='primary')
        self.remove_button.on_click(self.clean)
        
        
    
    def clean(self, event):
        print('cleaning')
        stop_words_cleaner = StopWordsCleaner() 
        stop_words_cleaner.setInputCols(["token"])
        stop_words_cleaner.setOutputCol("cleanTokens")
        stop_words_cleaner.setCaseSensitive(False)  #You may or may not care about case.
        stop_words_cleaner.setStopWords(self.stopwords)
            
        clean_token_df=stop_words_cleaner.transform(self.spark_df)
        
        self.spark_df = clean_token_df
        

        self.display_df = get_all_lines(self.spark_df, 'cleanTokens.result', col = 'cleanTokens')
        #now that cleaning has been done, can continue again
        self.continue_button.disabled = False
        

    def options_page(self):
        
        return pn.WidgetBox(self.load_words_button,
                            self.remove_button,
                height = 300,
                width = 300
        
        )
    
    @param.output('spark_df', 'display_df')
    def output(self):
        return self.spark_df, self.display_df