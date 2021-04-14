import panel as pn
import param
import pandas as pd

pn.extension()

class base_page(param.Parameterized):
    
    name_of_page = param.String(default = 'Name of page')
    
    display_df = param.DataFrame(default = pd.DataFrame())
    
    ready = param.Boolean(
        default=False,
        doc='trigger for moving to the next page',
        )   
    
    def __init__(self, **params):
        
        super().__init__(**params)
        self.continue_button = pn.widgets.Button(name='Continue', 
                                                 disabled = True,
                                                 width = 100,
                                                 button_type='primary')
    
        self.continue_button.on_click(self.continue_ready)
    
    def continue_ready(self, event):
        self.ready=True

        
    def options_page(self):
        return pn.WidgetBox(
                height = 300,
                width = 300
        
        )
        
    @param.depends('display_df')
    def df_pane(self):
        return pn.WidgetBox(self.display_df,
                           height = 300,
                           width = 400)
    
    def continue_button_pane(self):
        return pn.Row(pn.Spacer(width = 290), self.continue_button)
        
        
    def panel(self):
        
        return pn.Row(
                pn.Column(
                    pn.pane.Markdown(f'##{self.param.name_of_page.default}'),
                    self.options_page,
                ),
                pn.Column(
                    pn.Spacer(height=52),
                    self.df_pane,
                    self.continue_button_pane,
                    
                )
        
        )
        
        
def get_all_lines(spark_df, col_val, col = None):
    cnt = spark_df.count()
    lines = spark_df.select(f'{col_val}').take(cnt)
    labels = spark_df.select('sentiment').take(cnt)

    for n in range(cnt):
        lines[n] = lines[n][0]
        labels[n] = labels[n][0]
        
    if col == None:
        col_name = col_val
    else:
        col_name = col
    return pd.DataFrame({f'{col_name}': lines, 'sentiment': labels})


def join_lines(df):
    cnt = len(df)

    lines = []
    for n in range(cnt):
        lines.append(' '.join(df.stem[n]))

    return lines