import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

@st.cache_data
def load_data(nrows):
    df = pd.read_excel("data/Pondy_Analysis_New.xlsx")
    df = df.iloc[:nrows,1:]
    df = df[df["Status"]=="Success"]
    return df

st.header("Demand Data Analysis of Plant 2")

#data_loading_state = st.text("Loading data...")

#data_loading_state.text("Done! (using st.cache_data)")

sample_data = load_data(11000)

bar_data = pd.DataFrame({
    "Category": ["Low Value Goods" , "Medium Value Goods" , "High Value Goods"],
    "Count":[
        len(sample_data[sample_data["Price"] <= 10000]),
        len(sample_data[(sample_data["Price"] <= 50000) & (sample_data["Price"] > 10000)]),
        len(sample_data[sample_data["Price"] > 50000])
    ],
    "Price Range": ["≤ ₹10,000", "₹10,001 – ₹50,000", "> ₹50,000"]
})

price_fig = px.bar(
    bar_data,
    x="Category",
    y="Count",
    color="Category",
    hover_data="Price Range",
    title="Distribution of Goods by Value",
    color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
)

st.plotly_chart(price_fig)

adi_plot = pd.DataFrame({
    "Category":["Fast Movers" , "Medium Movers" , "Slow Movers"],
    "Count":[
        len(sample_data[sample_data["ADI"] <= 2]),
        len(sample_data[(sample_data["Price"] <= 5) & (sample_data["Price"] > 2)]),
        len(sample_data[sample_data["Price"] > 5])
    ],
    "ADI Range": ["<=2", "2-5", ">5"]
})

adi_fig = px.bar(
    adi_plot,
    x="Category",
    y="Count",
    color="Category",
    hover_data="ADI Range",
    title="Distribution of Goods by Average Demand Interval",
    color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
)

st.plotly_chart(adi_fig)
gof_plant_2 = [
    {
        "Model":"ZISPD",
        "% SKU showing Strong Fit": "99.7%"
    },
    {
        "Model":"ZINB",
        "% SKU showing Strong Fit": "99.2%"
    },
    {
        "Model":"ZIP",
        "% SKU showing Strong Fit": "96.89%"
    },
    {
        "Model":"ZIGP",
        "% SKU showing Strong Fit": "70.4%%"
    },
    {
        "Model":"Poisson",
        "% SKU showing Strong Fit": "63.8%"
    }

]

df = pd.DataFrame(gof_plant_2)
st.dataframe(df , hide_index=True , use_container_width=True)


