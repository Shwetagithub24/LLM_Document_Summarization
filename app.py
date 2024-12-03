import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration # Conditional generation for Model loading
from transformers import pipeline
import torch
import base64 # to display pdf on streamlit
from rouge import rouge_score


#load model and tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")

#Load file and preprocess
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap = 50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content
    return final_texts

# LLM Pipeline
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50,
        #do_sample=False,  # enforces deterministic output
        #top_k=30, #controlling randomness
        #top_p=0.95, #allows dynamic adjustment to include only the most relevant tokens.
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result


@st.cache_data
def displayPDF(file):
    #Open file from path
    with open(file,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    #Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    
    #Displaying File
    st.markdown(pdf_display,unsafe_allow_html=True)


#Streamlit code
st.set_page_config(layout='wide', page_title="Summarization Application")

def main():
    st.title('PDF Summarizer using Large Language Model')
    st.write('This tool can summarize large PDF files into concise summaries.')

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
    if uploaded_file is not None:
        if st.button('Summarize'):
            col1,col2 = st.columns(2)
            filepath = "data/"+uploaded_file.name
            with open(filepath, 'wb') as temp_file:
                temp_file.write(uploaded_file.read())

            with col1:
                st.info("Uploaded PDF File.")
                pdf_viewer = displayPDF(filepath)
            with col2:
                st.info("Summarization is Completed.")
                
                summary = llm_pipeline(filepath)
                st.success(summary)

    

    def evaluate_summary(reference_text, generated_text):
        reference_text = "The U.S. Department of Agriculture's Animal Care Aid discusses Canine Periodontal Disease (PD), which is inflammation of tissues and bone around the teeth due to a bacterial infection. PD develops from plaque hardening into tartar along the gumline, leading to gum inflammation and potentially severe consequences like pain, infection, abscesses, and tooth loss. Risk factors for PD include breed size, muzzle length, age, and breed. Signs of advanced PD include bad breath, excessive drooling, difficulty chewing, and weight loss. PD in dogs can lead to significant health and welfare problems, such as kidney, liver, and heart disease, bleeding gums, tooth loss, and pain. Email CenterforAnimalWelfare@aphis.usda.gov for questions."
        generated_text = summary

        scores = evaluate_summary(reference_text, generated_text)
        print("Evaluation Scores:", scores)
        

if __name__ == '__main__':
    main()
