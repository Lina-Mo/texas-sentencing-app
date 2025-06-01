import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("Texas_Department_of_Criminal_Justice_Receives_FY_2024.csv")

# Clean sentence years
def convert_sentence_to_years(text):
    if pd.isna(text): return None
    text = text.lower()
    if "year" in text:
        return float(''.join([c for c in text if c.isdigit() or c == '.']))
    elif "month" in text:
        return round(float(''.join([c for c in text if c.isdigit() or c == '.'])) / 12, 2)
    elif "life" in text:
        return 99
    elif "death" in text:
        return 100
    else:
        return None

df['Sentence_Years_Numeric'] = df['Sentence Years'].apply(convert_sentence_to_years)

st.title("📊 Data Explorer – Texas Criminal Justice")

# Tabs for layout
tab1, tab2, tab3 = st.tabs(["Demographics", "Offense Trends", "Sentencing"])

with tab1:
    st.header("🔹 Gender & Race Distribution")

    col1, col2 = st.columns(2)
    with col1:
        gender_counts = df['Gender'].value_counts()
        st.bar_chart(gender_counts)

    with col2:
        race_counts = df['Race'].value_counts()
        st.bar_chart(race_counts)

with tab2:
    st.header("🔹 Top Offenses")
    top_offenses = df['Offense'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_offenses.values, y=top_offenses.index, ax=ax)
    ax.set_title("Top 10 Offenses")
    ax.set_xlabel("Count")
    st.pyplot(fig)

with tab3:
    st.header("🔹 Sentence Analysis by Race")
    race_sent = df[['Race', 'Sentence_Years_Numeric']].dropna()
    fig, ax = plt.subplots()
    sns.boxplot(data=race_sent, x='Race', y='Sentence_Years_Numeric', ax=ax)
    ax.set_title("Sentence Length Distribution by Race")
    ax.set_ylabel("Years")
    st.pyplot(fig)

    st.header("🔹 Sentence Analysis by Offense")
    offense_sent = df[df['Offense'].isin(df['Offense'].value_counts().head(10).index)]
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=offense_sent, x='Offense', y='Sentence_Years_Numeric', ax=ax2)
    ax2.set_title("Sentence Length by Top 10 Offenses")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig2)
