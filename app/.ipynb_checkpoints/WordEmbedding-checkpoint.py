from sparknlp.annotator import WordEmbeddingsModel
from app.base import base_page
from app.base import get_all_lines, join_lines
import panel as pn
import pandas as pd
import param
from pyspark.sql.dataframe import DataFrame as sdf


class WordEmbedder(base_page):
    
    spark_df = param.ClassSelector(
        class_= sdf
    )
    
    display_df = param.DataFrame(default = pd.DataFrame())
    
    df = param.DataFrame()
    X = param.Array(default = None)
    
    def __init__(self, **params):
        super().__init__(**params)
#         self.spark_df = spark_df
        
        self.param.name_of_page.default = 'Word Embedding'
        self.we_model = pn.widgets.Select(name='Select', options=['SKLearn Count Vectorizer', 'Glove', 'Bert'])

        self.we_button = pn.widgets.Button(name='Transform', button_type='primary')
        self.we_button.on_click(self.transform)
        
    def options_page(self):
        
        return pn.WidgetBox(self.we_model,
                            self.we_button,
                height = 300,
                width = 300
        
        )
    
    def transform(self, event):
        print('embedding')
        
        if self.we_model.value == 'Glove':
            print('glove')
            from sparknlp.annotator import WordEmbeddingsModel
            word_embeddings=WordEmbeddingsModel.pretrained()
            word_embeddings.setInputCols(['document','stem'])
            word_embeddings.setOutputCol('embeddings')

            self.spark_df = word_embeddings.transform(self.spark_df)
            
            embeddings_df = get_all_lines(self.spark_df, 'embeddings.embeddings', col = 'embeddings')
            
        if self.we_model.value == 'SKLearn Count Vectorizer':
            from sklearn.feature_extraction.text import CountVectorizer
            print('join lines')
            corpus = join_lines(self.display_df)
            print('doing vectorizer')
            vectorizer = CountVectorizer(max_features=1500)
            print('vectorizing 2')
            X = vectorizer.fit_transform(corpus).toarray()

            cnt = self.spark_df.count()
            print('getting sentiment from spark df')
            labels = self.spark_df.select('sentiment').take(cnt)

            for n in range(cnt):
                labels[n] = labels[n][0]
            print('done getting sentiment, creating dataframe')
            xlist = []
            for n in range(len(X)):
                xlist.append(list(X[n]))
            self.X = X
            embeddings_df = pd.DataFrame({'embeddings': xlist, 'sentiment': labels})
        
        else: 
            print('bert')
            from sparknlp.annotator import BertEmbeddings
            bertEmbeddings = BertEmbeddings.pretrained()
            
            bertEmbeddings.setInputCols(['document','stem'])
            bertEmbeddings.setOutputCol('embeddings')

            embeddings_df=bertEmbeddings.transform(self.spark_df)
        
            self.spark_df = embeddings_df
            
            embeddings_df = get_all_lines(self.spark_df, 'embeddings.embeddings', col = 'embeddings')
        

        self.display_df = embeddings_df
        self.continue_button.disabled = False
    
    
    @param.output('X', 'display_df')
    def output(self):
        return self.X, self.display_df
        
        
        