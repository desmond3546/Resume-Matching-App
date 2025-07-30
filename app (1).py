import streamlit as st
import pickle
import docx
import PyPDF2
import re
import numpy as np

svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', ' ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText.strip()

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        raise ValueError("Could not extract text from the PDF. It may be encrypted or corrupted.")

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')

def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload PDF, DOCX, or TXT.")

def predict_resume_category(text):
    cleaned = cleanResume(text)
    vect_text = tfidf.transform([cleaned])
    vect_array = vect_text.toarray()
    prediction = svc_model.predict(vect_array)
    predicted_label = le.inverse_transform(prediction)[0]

    try:
        proba = svc_model.predict_proba(vect_array)[0]
        top_indices = np.argsort(proba)[::-1][:3]
        top_classes = le.inverse_transform(top_indices)
        top_probs = proba[top_indices]
        return predicted_label, list(zip(top_classes, top_probs))
    except:
        return predicted_label, None

def main():
    st.set_page_config(page_title="Resume Category Classifier", layout="wide")

    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume (`.pdf`, `.docx`, or `.txt`) and the model will predict the **most likely job category**.")

    st.sidebar.header("Options")
    show_text = st.sidebar.checkbox("Show Extracted Text", False)
    max_file_size_mb = st.sidebar.slider("Max File Size (MB)", 1, 10, 3)

    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        file_size = uploaded_file.size / (1024 * 1024)  # in MB
        if file_size > max_file_size_mb:
            st.error(f"File size exceeds {max_file_size_mb}MB limit. Please upload a smaller file.")
            return

        try:
            resume_text = handle_file_upload(uploaded_file)
            if not resume_text.strip():
                st.error("The uploaded file doesn't contain readable text.")
                return

            st.success("Successfully extracted text from the resume.")

            if show_text:
                st.text_area("Extracted Resume Text", resume_text, height=300)

            with st.spinner("Predicting category..."):
                label, top_preds = predict_resume_category(resume_text)

            st.markdown(f"### Predicted Category: **{label}**")

            if top_preds:
                st.markdown("#### Top Predictions:")
                for cat, prob in top_preds:
                    st.write(f"• **{cat}** — {prob:.2%} confidence")

        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.markdown("---")
    st.info("This app uses a Random Forest classifier trained on resume text for job role classification.")

if __name__ == "__main__":
    main()
