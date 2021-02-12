from app.base import base_page
import panel as pn
import pandas as pd
import param
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np


class trainer(base_page):
    
    display_df = param.DataFrame(default = pd.DataFrame())
    
    results = param.Boolean(default = False)
    
    X = param.Array(default = None)
    
    result_string = param.String(default = '')

    result_string = param.String('')
    
    def __init__(self, **params):
        super().__init__(**params)
        self.param.name_of_page.default = 'Test and Train'
        
        self.test_slider = pn.widgets.IntSlider(name='Test Percentage', start=0, end=100, step=10, value=20)

        self.tt_button = pn.widgets.Button(name='Train and Test', button_type='primary')
        self.tt_button.on_click(self.train_test)
        
        self.tt_model = pn.widgets.Select(name='Select', options=['Random Forrest Classifier'])
        
        print(self.display_df)
        
    def train_test(self, event):
        
        #get values from sentiment.
        self.display_df = convert_sentiment_values(self.display_df)
        
        y = self.display_df['label']
        
        #get train test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, y, test_size = self.test_slider.value/100, random_state = 0)
        
        
        if self.tt_model.value == 'Random Forrest Classifier':
            sentiment_classifier = RandomForestClassifier(n_estimators = 1000, random_state = 0)
            
            sentiment_classifier.fit(X_train, y_train)
            
            y_pred = sentiment_classifier.predict(X_test)
            print(y_pred)
            
        self.y_test = y_test
        self.y_pred = y_pred
        self.analyze()
        
    def analyze(self):
        self.cm = confusion_matrix(self.y_test,self.y_pred)
        self.cr = classification_report(self.y_test,self.y_pred)
        self.acc_score = accuracy_score(self.y_test, self.y_pred)
        
        splits = self.cr.split('\n')
        cml = self.cm.tolist()
        self.result_string = f"""
            ### Classification Report
            <pre>
            {splits[0]}
            {splits[1]}
            {splits[2]}
            {splits[3]}
            {splits[4]}
            {splits[5]}
            {splits[6]}
            {splits[7]}
            {splits[8]}
            </pre>
            ### Confusion Matrix
            <pre>
            {cml[0]}
            {cml[1]}

            </pre>

            ### Accuracy Score
            <pre>
            {round(self.acc_score, 4)}
            </pre
            """
        

        self.results = True 

    def options_page(self):
        
        return pn.WidgetBox(self.tt_model,
                            self.test_slider,
                            self.tt_button,
                height = 300,
                width = 300
        
        )
        
    @pn.depends('results')
    def df_pane(self):
        
        if self.results == False:
            self.result_pane = self.display_df
            
        else:
            self.result_pane = pn.pane.Markdown(f"""
                {self.result_string}
                """, width = 500, height = 350)
        
        return pn.WidgetBox(self.result_pane,
                           height = 375,
                           width = 450)
        


    def panel(self):
        
        return pn.Row(
                pn.Column(
                    pn.pane.Markdown(f'##{self.param.name_of_page.default}'),
                    self.options_page,
                ),
                pn.Column(
                    pn.Spacer(height=52),
                    self.df_pane,
                    
                )
        
        )
    
    
def convert_sentiment_values(df, col = 'sentiment'):
    vals = df['sentiment'].unique()
    df['label'] = 0

    for n in range(len(vals)):
        df['label'] = [n if df[col][x] == vals[n] else df['label'][x] for x in range(len(df[col]))]
        
    return df




    