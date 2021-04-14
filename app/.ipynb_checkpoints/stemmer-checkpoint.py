from sparknlp.annotator import Stemmer
from app.base import base_page
from app.base import get_all_lines
import param
import panel as pn
import pandas as pd
from pyspark.sql.dataframe import DataFrame as sdf


class App_Stemmer(base_page):
    
    spark_df = param.ClassSelector(
        class_= sdf
    )
    display_df = param.DataFrame(default = pd.DataFrame())
    
    
    def __init__(self, **params):
        super().__init__(**params)
        self.param.name_of_page.default = 'Stemming/Lemmatizing'
        
        self.stem_button = pn.widgets.Button(name='Stem', button_type='primary')
        self.stem_button.on_click(self.stem)
        
        self.lem_button = pn.widgets.Button(name='Lemmatize', disabled = True, button_type='primary')
        
    def stem(self, event):
        stemmer = Stemmer()
        self.param.name_of_page.default = 'Stemming'
        
        stemmer.setInputCols(["cleanTokens"]) 
        stemmer.setOutputCol("stem")

        stem_df=stemmer.transform(self.spark_df)
        
        self.spark_df = stem_df
        self.display_df = get_all_lines(self.spark_df, 'stem.result', col ='stem')
    
        self.continue_button.disabled = False
        
        
    def options_page(self):
        
        return pn.WidgetBox(self.stem_button,
                            self.lem_button,
                height = 300,
                width = 300
        
        )
        
    @param.output('spark_df', 'display_df')
    def output(self):
        return self.spark_df, self.display_df