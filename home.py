import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.title("Fitting Probabilistic Models on Demand Distribution of Spare Items")

st.set_page_config(layout="wide")

st.header("Models Considered for Spare Part Demand")

c1,c2 = st.columns(2)

st.markdown("""
    <style>
    
    div[data-testid="stVerticalBlockBorderWrapper"] > div:first-child {
        overflow-y: auto !important;
    }
    </style>
    """, unsafe_allow_html=True)

with c1:
    with st.container(border=True, height=400): # Using border and height for a card effect
        st.header("Poisson Distribution")
        st.subheader("Formula")
        st.latex(r'''
            P(X=x) = \frac{\lambda^x e^{-\lambda}}{x!}
            ''')
        st.subheader("Strengths:")
        st.markdown("""
            1. It is a discrete distribution, which helps us capture the countable nature of demand data.
            2. It has a single parameter (λ) representing the average demand rate, making it computationally efficient.
            """)
        st.subheader("Limitations")
        st.markdown("""
        1.Not able to capture the zero inflation rate efficiently which is the majority of the data in intermittent demand.
        """)

    with st.container(border=True, height=400):
        st.header("Zero Inflated Poisson")
        st.subheader("Formula")
        st.latex(r'''
            P(X=x) = 
            \begin{cases} 
            \pi + (1-\pi)e^{-\lambda} & \text{if } x=0 \\
            \\
            (1-\pi)\frac{\lambda^x e^{-\lambda}}{x!} & \text{if } x > 0 
            \end{cases}
            ''')
        st.subheader("Strengths:")
        st.markdown("""
                    1.Is able to capture the zeros better than the simple poisson distribution because of the parameter Pi(π).

                    2. Is still a relatively simpler model which is not too computationally intensive.
                    """)
        st.subheader("Limitations")
        st.markdown("""
                Since the assumption of poisson distribution is that Mean = Variance, so the ZIP model is not able to capture the overdispersed and underdispersed data.
                """)

with c2:
    with st.container(border=True, height=400):
        st.header("Zero Inflated Generalized Poisson")
        st.latex(r'''
            P(X=k) = 
            \begin{cases} 
            \pi + (1-\pi)e^{-\lambda} & \text{if } x=0 \\
            \\
            (1-\pi)\frac{\lambda(\lambda+x\phi)^{x-1}}{x!} e^{-(\lambda+k\phi)} & \text{if } x > 0 
            \end{cases}
            ''')
        st.subheader("Strengths:")
        st.markdown("""Is able to capture the overdispersed or the underdispersed data better than Poisson or ZIP model.""")
        st.subheader("Limitations")
        st.markdown("""1. Computationally intensive with multiple parameters.
                       2. Underperforms when the number of data points are limited.""")

    with st.container(border=True, height=400):
        st.header("Zero Inflated Negative Binomial")
        st.latex(r'''
            \text{P}(X=x) = 
            \begin{cases} 
            \pi + (1-\pi)\left(\frac{\Theta}{\mu + \Theta}\right)^\Theta & \text{if } x=0 \\
            \\
            (1-\pi) \frac{\Gamma(x+\Theta)}{\Gamma(\Theta)x!} \left(\frac{\mu}{\mu + \Theta}\right)^x & \text{if } x > 0
            \end{cases}
            ''')

        st.subheader("Strengths:")
        st.markdown("""
                        1. Better at dealing with overdispersed data tha ZIGP.
                        
                        2. Less volatile than ZIGP""")
        st.subheader("Limitations")
        st.markdown("""1. Performs poorly on underdispersed data and is computationally intensive.""")
    with st.container(border=True, height=400):
        st.header("Stuttering Poisson Distribution")
        st.subheader("Formula (Probability Generating Function)")
        st.latex(r'G_X(s) = e^{\lambda\Sigma\alpha(s-1)}')

        st.subheader("Strengths:")
        st.markdown("""1. The geometric distribution in this model helps to capture the bursts of demands sizes among the other zero demand values.""")
        st.markdown(""" 2. Is the best fitting model compared to all the other models.""")
        st.subheader("Limitations")
        st.markdown("""1. Large number of parameters can overfit on smaller data.""")
        st.markdown(""" 2. The PMF of SPD is recursive in nature so is very computationally intensive""")


st.header("2. Parameter Estimation: Maximum Likelihood Estimation (MLE)")
st.markdown("""For Parameter estimation, MLE is used where the Likelihood for a given distribution for a given set of parameters is given by:""")
st.latex(r'''
    L(\theta | \mathbf{x}) = \prod_{i=1}^{n} f(x_i | \theta)
    ''')
st.markdown("""To make it computationally efficient, the multiplicaitve series is converted to additive series by taking Log of the Likelihood function which is given by:""")

st.latex(r'''
    \mathcal{L}(\theta | \mathbf{x}) = \ln L(\theta | \mathbf{x}) = \sum_{i=1}^{n} \ln f(x_i | \theta)
    ''')
st.markdown("""This Log-Likelihood is then maximized to obtain the best set of parameters""")


st.header("3. Goodness of Fit: Kolmogorov-Smirnov(KS) Test")
st.markdown("""The KS Test compares the empirical cdf distribution to the theoritical cdf distribution and finds the maximum vertical distance between the two distributions.""")
st.latex(r'''
    D_n = \sup_{x} \left| F_n(x) - F(x) \right|
    ''')
st.markdown("""Where D signifies the KS statistic, F_n signifies the Empirical CDF and F(X) signifies the theoritical cdf distribution""")

st.header("4. Relative Model Comparision")

fit = [
    {
        "Range of KS Statistic": "Less than the critical value at 5% significance level",
        "Type of Fit": "STRONG FIT"
    },
    {
        "Range of KS Statistic": "Between the critical values of 1% and 5% significance level",
        "Type of Fit": "POOR FIT"
    },
    {
        "Range of KS Statistic": "Greater than the critical value at 1% significance level",
        "Type of Fit": "NO FIT"
    }
]

df = pd.DataFrame(fit)
st.dataframe(df, hide_index=True , use_container_width=True)
