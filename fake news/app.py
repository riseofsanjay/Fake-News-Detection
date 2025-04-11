import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import base64

# Load the model and vectorizer
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.title("📰 Fake News Detector")
st.write("Enter a news article below to check whether it is Fake or Real.")

# Input text
inputn = st.text_area("✍️ News Article:", "")

if st.button("🔍 Check News"):
    if inputn.strip():
        transform_input = vectorizer.transform([inputn])
        prediction = model.predict(transform_input)
        confidence = model.predict_proba(transform_input)  # Get confidence scores

        confidence_score = np.max(confidence) * 100  # Convert to percentage

        if prediction[0] == 1:
            st.success(f"✅ The News is **Real**! (Confidence: {confidence_score:.2f}%)")
        else:
            st.error(f"❌ The News is **Fake**! (Confidence: {confidence_score:.2f}%)")

        # Extract important words
        feature_names = vectorizer.get_feature_names_out()
        important_words = np.argsort(transform_input.toarray()[0])[-10:]  # Top 10 words
        keywords = [feature_names[i] for i in important_words]

        # Display Keywords
        st.subheader("🔍 Important Keywords Identified:")
        highlighted_text = inputn
        for word in keywords:
            highlighted_text = highlighted_text.replace(word, f"**{word}**")
        st.markdown(highlighted_text)

        # Generate and display WordCloud
        st.subheader("📊 WordCloud of Important Words")
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(keywords))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Bar Chart of Word Importance
        st.subheader("📈 Word Importance Bar Chart")
        word_importance = {feature_names[i]: transform_input.toarray()[0][i] for i in important_words}
        word_importance = dict(sorted(word_importance.items(), key=lambda x: x[1], reverse=True))

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=list(word_importance.values()), y=list(word_importance.keys()), ax=ax, palette="coolwarm")
        ax.set_xlabel("Word Frequency/Impact")
        ax.set_ylabel("Top Words")
        st.pyplot(fig)

        # Downloadable Analysis Report
        st.subheader("📄 Download Analysis Report")
        report_text = f"Fake News Detection Report\n\nNews Article:\n{inputn}\n\nPrediction: {'Real' if prediction[0] == 1 else 'Fake'}\nConfidence: {confidence_score:.2f}%\n\nImportant Keywords:\n{', '.join(keywords)}"
        
        b64 = base64.b64encode(report_text.encode()).decode()  # Convert report to Base64
        href = f'<a href="data:file/txt;base64,{b64}" download="FakeNewsAnalysis.txt">📥 Download Report</a>'
        st.markdown(href, unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please enter some text to analyze.")
