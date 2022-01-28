pip install scikit-learn

import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = 'Sales_clustering_6.xlsx'


@st.cache
def load_data(path):
    data = pd.read_excel(path)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

data_load_state = st.text('Loading data...')
df = load_data(DATA_PATH)

data_load_state.text("Done loading data!")

X = df[['cost price', 'selling price', 'quantity']]
Y = df['if profit']

@st.cache
def agg_data(X, mode):
    dat = X.agg([mode])
    return dat

data_agg_state = st.text('Aggregating data...')
dfMin = agg_data(X, 'min')
dfMax = agg_data(X, 'max')
dfMedian = agg_data(X, 'median')
dfMode = agg_data(X, 'mode')
data_agg_state.text("Done aggregating data!")

st.title('Predict Profit or Loss')
st.sidebar.title("Features")

parameter_input_values=[]
values=[]

for parameter in X:
	values = st.sidebar.slider(label=parameter, key=parameter, value=float(dfMedian[parameter]), min_value=float(dfMin[parameter]), max_value=float(dfMax[parameter]), step=0.1)
	parameter_input_values.append(values)

parameter_list = X.columns
input_variables=pd.DataFrame([parameter_input_values],columns=parameter_list)
st.write('\n\n')

model = DecisionTreeClassifier()
model.fit(X, Y)

from PIL import Image
yes = Image.open('yes.jpg').resize((200, 300))
no = Image.open('no.jpg').resize((200, 300))

if st.button("Will the it be a profit?"):
    prediction = model.predict(input_variables)
    #pred = 'No' if prediction == 0 else 'Yes'
    #st.text(pred)
    img = no if prediction == 0 else yes
    st.image(img)
