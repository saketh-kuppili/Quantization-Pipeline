import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from quant_pipeline.core.pipeline import Pipeline

st.title("Quantization Dashboard")

text = st.text_input("Input text")
mode = st.selectbox("Precision",["fp32","fp16","int8_ptq","int8_qat"])

if st.button("Predict"):
    pipe = Pipeline(precision=mode)
    st.write(pipe.predict(text))

try:
    df = pd.read_csv("outputs/results.csv")
    st.dataframe(df)

    fig, ax = plt.subplots()
    ax.bar(df["mode"], df["accuracy"])
    st.pyplot(fig)

except:
    st.warning("Run benchmark first")