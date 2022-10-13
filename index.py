from multiprocessing.sharedctypes import Value
import dash

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dash import Dash, dcc, html, Input, Output
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
from sklearn import metrics, datasets
# Import Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, plot_confusion_matrix, plot_precision_recall_curve




app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])

url='assets/heart.csv'
df = pd.read_csv(url)
df_model= pd.read_csv(url)
df.isna().sum()
df_cor=df.corr()
fig_heat= px.imshow(df_cor)
label_dict = {1: 'Heart Disease', 0: 'Normal'}
df['HeartDisease'] = df['HeartDisease'].map(label_dict)

cat_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'HeartDisease']
Numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in cat_features:
    df_model[i] =  le.fit_transform(df_model[i])

X = df_model.drop(columns = ['HeartDisease'])
Y = df_model[['HeartDisease']]

from sklearn.preprocessing import StandardScaler
sc =  StandardScaler()
scaled_data = sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 3)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

from sklearn.linear_model import LogisticRegression
lr =  LogisticRegression()
lr.fit(X_train, Y_train)

# logistic regression
lr_predict = lr.predict(X_test)
print(accuracy_score(Y_test, lr_predict))

# support vector machine
from sklearn import svm
SVM = svm.SVC()
SVM.fit(X_train, Y_train)

SVM_predict = SVM.predict(X_test)
print(accuracy_score(Y_test, SVM_predict))

# catboost machine

from catboost import CatBoostClassifier
cbc =  CatBoostClassifier()
cbc.fit(X_train, Y_train)

cbc_predict =  cbc.predict(X_test)
print(accuracy_score(Y_test, cbc_predict))


label_dict = {1: 'High Fasting Blood Sugar', 0: 'Low Fasting Blood Sugar'}
df['FastingBS'] = df['FastingBS'].map(label_dict)
def cholestrol(x):
    if x >= 500:
        return " Very High" 
    if x<= 499 and x >= 200:
        return "High"
    elif x<= 199 and x >= 150:
        return "Mildly High"
    else:
        return "Low"


df["Cholesterol"] = df["Cholesterol"].apply(cholestrol)
def MaxHR(x):
    if x >= 500:
        return " Very High" 
    if x<= 180 and x >= 141:
        return "High"
    elif x<=140  and x >= 120:
        return "Mildly High"
    else:
        return "Low"
    

df["MaxHR"] = df["MaxHR"].apply(MaxHR)

def Oldpeak(x):
   
         
    if  x >= 4.01:
        return "High"
    elif x<= 4 and x >= 1:
        return "Mild"
    else:
        return "Low"
    

df["Oldpeak"] = df["Oldpeak"].apply(Oldpeak)
def Age(x):
   
    if  x >50:  
        return "Above 50"
    elif  x<= 50 and x >= 41:
        return "41 - 50"
    elif x<= 40 and x >= 30:
        return "30 - 40"
    else:
        return "Under 30"
    

df["Age"] = df["Age"].apply(Age)

def resting_bp(x):
   
    if  x >121:  
        return "Very High"
    elif  x<= 120 and x >= 101:
        return "High"
    elif x<= 100 and x >= 60:
        return "Normal"
    else:
        return "Low"
    

df["RestingBP"] = df["RestingBP"].apply(resting_bp)

fig = px.histogram(df, x="HeartDisease",color="HeartDisease",pattern_shape="HeartDisease")

score = df['HeartDisease']
subject = df['Age']

data = [dict(
  type = 'histogram',
  y = subject,
  x = score,
  mode = 'markers',
  transforms = [dict(
    type = 'groupby',
    groups = subject,
    styles = [
        
    ]
  )]
)]

fig_dict = dict(data=data)

# Categorical Features
cat_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'HeartDisease']
Numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for i in cat_features:
    df_model[i] =  le.fit_transform(df_model[i])

X = df_model.drop(columns = ['HeartDisease'])
Y = df_model[['HeartDisease']]

df_model[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']].corr()

MODELS = {'Logistic Regression': linear_model.LogisticRegression,
          'Decision Tree': tree.DecisionTreeClassifier,
          'k-NN': neighbors.KNeighborsClassifier}

          

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Img(src=app.get_asset_url('download.jpg'),
                     id='corona-image',
                     style={
                         "height": "90px",
                         "width": "auto",
                         "margin-bottom": "25px",
                     },     
                     )
        ],
        
            className="one-third column",
        ),
        html.Div([
            html.Div([
                html.H3("", style={"margin-bottom": "0px", 'color': 'white'}),
                html.H5("Heart Failure Analysis and Prediction", style={"margin-top": "0px", 'color': 'White'}),
                html.Img(src=app.get_asset_url('ekg-heart-rate.gif'),
                     id='heart-image',
                     style={
                         "height": "100px",
                         "width": "auto",
                         "margin-bottom": "25px",
                     },)
            ])
        ], className="one-third column", id="title" ),


    ], id="header", className="row flex-display", style={"margin-bottom": "25px"}),

    html.Div([
        html.Div([
            html.H6(children='Logistic Regression',
                    style={
                        'textAlign': 'center',
                        'color': 'White'}
                    ),

            html.P(f"{accuracy_score(Y_test, lr_predict)}",
                   style={
                       'textAlign': 'center',
                       'color': 'White',
                       'fontSize': 30}
                   ),

            html.P("Model Accuracy",
                   style={
                       'textAlign': 'center',
                       'color': 'White',
                       'fontSize': 18,
                       'margin-top': '-18px'}
                   )], className="card_container four columns",
        ),

        html.Div([
            html.H6(children='Support Vector Machine',
                    style={
                        'textAlign': 'center',
                        'color': 'White'}
                    ),

            html.P(f"{accuracy_score(Y_test, SVM_predict)}",
                   style={
                       'textAlign': 'center',
                       'color': '#dd1e35',
                       'fontSize': 30}
                   ),

            html.P("Model Accuracy",
                   style={
                       'textAlign': 'center',
                       'color': '#dd1e35',
                       'fontSize': 18,
                       'margin-top': '-18px'}
                   )], className="card_container four columns",
        ),

        html.Div([
            html.H6(children='Catboost Machine',
                    style={
                        'textAlign': 'center',
                        'color': 'White'}
                    ),

            html.P(f"{accuracy_score(Y_test, cbc_predict)}",
                   style={
                       'textAlign': 'center',
                       'color': '#7fd37f',
                       'fontSize': 30}
                   ),

            html.P("Model Accuracy",
                   style={
                       'textAlign': 'center',
                       'color': '#7fd37f',
                       'fontSize': 18,
                       'margin-top': '-18px'}
                   )], className="card_container four columns")

    ], className="row flex-display"),


    html.Div([

                    html.P('Filter Features:', className='fix_label',  style={'color': 'white'}),

                    dcc.Dropdown(df.columns, id='pandas-dropdown-1', value= 'HeartDisease'),               

        ], className="create_container six columns", id="cross-filter-options"),
        
    html.Div([
        
            html.Div([
                      html.P('Distribution of Data', className='fix_label',  style={'color': 'white'}),
                      dcc.Graph(figure=fig,id='my-graph'),
                              ], className="create_container six columns"),

                    html.Div([
                        html.P('Important Relationships', className='fix_label',  style={'color': 'white'}),
                        dcc.Graph(figure=fig_dict, id='relation-graph'),
                        

                    ], className="create_container six columns"),

        ], className="row flex-display"),


html.Div([
        html.Div([
            html.P('Correlations Matrix', className='fix_label',  style={'color': 'white'}),
            dcc.Graph(figure=fig_heat, id="graph-heat")], className="create_container1 twelve columns"),

            ], className="row flex-display"),

html.Div([
        
            html.Div([
                      html.P('Visualization with Pie Chart', className='fix_label',  style={'color': 'white'}),
                      html.P('Filter Features:', className='fix_label',  style={'color': 'white'}),
                      dcc.Dropdown(
                      id='my_dropdown',
                      options=df.columns,
                      multi=False,
                      clearable=False,
                      style={"width": "50%"},
                      value= 'HeartDisease',
                    ),
                    dcc.Graph(id='the_graph')   
                              ], className="create_container six columns", id="cross-filter-options-001"),

                    html.Div([
                        
                         html.P('Analysis of the ML models results using ROC and PR curves', className='fix_label',  style={'color': 'white'}),
                         html.P('Select model:', className='fix_label',  style={'color': 'white'}),
                         dcc.Dropdown(
                         id='dropdown-precision',
                         options=["Logistic Regression", "Decision Tree", "k-NN"],
                         value='Logistic Regression',
                         clearable=False
                        ),
                    dcc.Graph(id="fig_precision"),    

                    ], className="create_container six columns", id="cross-filter-options-002"),

        ], className="row flex-display"),

    ], id="mainContainer",
    style={"display": "flex", "flex-direction": "column"})


@app.callback(
    Output('my-graph', component_property='figure'),
    Input('pandas-dropdown-1', 'value')
)



    
def update_output(value):
   
    fig =  px.histogram(df, x=value, color=value)
    return fig

@app.callback(
    Output('relation-graph', component_property='figure'),
     Input('pandas-dropdown-1', 'value'),
)
def update_groupby(value):
    score = df[value]
    subject = df['HeartDisease']
     
    data = [dict(
    type = 'histogram',
    y = subject,
    x = score,
    mode = 'markers',
    transforms = [dict(
    type = 'groupby',
    groups = subject,
    styles = [
        
     ]
    )]
   )]

    fig_dict = dict(data=data)
    return fig_dict

@app.callback(
    Output(component_id='the_graph', component_property='figure'),
    [Input(component_id='my_dropdown', component_property='value')]
)

def update_graph(my_dropdown):
    dff = df

    piechart=px.pie(
            data_frame=dff,
            names=my_dropdown,
            hole=.3,
            
            )

    return (piechart)

@app.callback(
    Output("fig_precision", "figure"), 
    Input('dropdown-precision', "value"))
def train_and_display(model_name):
    cat_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'HeartDisease']
    Numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']


    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    for i in cat_features:
     df_model[i] =  le.fit_transform(df_model[i])

    X = df_model.drop(columns = ['HeartDisease'])
    Y = df_model[['HeartDisease']]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, random_state=42)

    model = MODELS[model_name]()
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    score = metrics.auc(fpr, tpr)
   
    fig_precision = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={score:.4f})',
        labels=dict(
            x='False Positive Rate', 
            y='True Positive Rate'))
    fig_precision .add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)

    return fig_precision 
# boxes

if __name__ == '__main__':
    app.run_server(debug=False)
