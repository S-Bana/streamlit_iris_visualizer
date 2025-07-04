import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


# function load dataset
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()


#‌ model setting
model = RandomForestClassifier()
model.fit(df.iloc[:,:-1], df['species'])

#‌ Streamlit design setting
st.sidebar.title("Input Features")
sepal_lenght = st.sidebar.slider("sepal length (cm)", float(df["sepal length (cm)"].min()), float(df["sepal length (cm)"].max()))
sepal_width = st.sidebar.slider("sepal width (cm)", float(df["sepal width (cm)"].min()), float(df["sepal width (cm)"].max()))
petal_lenght = st.sidebar.slider("petal length (cm)", float(df["petal length (cm)"].min()), float(df["petal length (cm)"].max()))
petal_width = st.sidebar.slider("petal width (cm)", float(df["petal width (cm)"].min()), float(df["petal width (cm)"].max()))
