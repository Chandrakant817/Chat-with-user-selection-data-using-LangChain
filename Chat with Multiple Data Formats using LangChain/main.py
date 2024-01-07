import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import easyocr
from tempfile import NamedTemporaryFile
import pandas as pd
import io
import openai
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


# Azure OpenAI credentials
deployment_name = "gpt-35-turbo"
os.environ["OPENAI_API_KEY"] = ""
os.environ['OPENAI_API_TYPE'] = 'azure'
os.environ['OPENAI_API_BASE'] = "https://chandrakantopenai.openai.azure.com/"
os.environ['OPENAI_API_VERSION'] = "2023-09-15-preview"

#st.image("Screenshot 2023-11-10 160350.png", width=500)

#st.image("celebal tech.png", width=500)
st.title("Question & Answering App")
st.sidebar.markdown("---")
# Sidebar with yellow background
st.sidebar.title("**Multiple Formats**")
st.sidebar.markdown("---")
#st.sidebar.image("pdf.png", width=75)
#uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
    
st.markdown(
    """
    <style>
        .footer {
            display: flex;
            justify-content: center;
            align-items: center;
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #FFA500;
            padding: 10px;
            margin-left: 120px;
        }
        .icon {
            display: inline-block;
            margin: 0 10px;
        }
    </style>
    <div class="footer">
        <span class="icon">
            <a href="https://www.linkedin.com/in/chandrakant-thakur-314414182?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank">
                <img src="https://img.icons8.com/color/48/000000/linkedin.png" style="width: 48px; height: 48px;"/>
            </a>
        </span>
        <span class="icon">
            <a href="https://instagram.com/thakur_chandrakant7?igshid=NGVhN2U2NjQ0Yg==" target="_blank">
                <img src="https://img.icons8.com/color/48/000000/instagram-new.png" style="width: 48px; height: 48px;"/>
            </a>
        </span>
        <span class="icon">
            <a href="https://www.youtube.com/@CelebalTechnologies" target="_blank">
                <img src="https://img.icons8.com/color/48/000000/youtube-play.png" style="width: 48px; height: 48px;"/>
            </a>
        </span>
        <span class="icon">
            <a href="https://github.com/Chandrakant817" target="_blank">
                <img src="https://img.icons8.com/ios-glyphs/30/000000/github.png" style="width: 48px; height: 48px;"/>
            </a>
        </span>
        <span class="icon">
            <a href="https://www.facebook.com/celebaltechnologies" target="_blank">
                <img src="https://img.icons8.com/color/48/000000/facebook-new.png" style="width: 48px; height: 48px;"/>
            </a>
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

    # Main content area with light yellow background
st.markdown(
        """
        <style>
            .main {
                background-color: #FFFFE0;
                padding: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True
)
st.markdown("<div class='main'>", unsafe_allow_html=True)

# CSV Function
def process_csv_question(csv_file, user_question):
    llm = AzureOpenAI(temperature=0, model_kwargs={'engine': 'gpt-35-turbo'})
    agent = create_csv_agent(llm, csv_file, verbose=True)
    st.write(agent.run(user_question))

# Excel function
def process_Excel_question(data, user_question):
    llm = AzureOpenAI(temperature=0, model_kwargs={'engine': 'gpt-35-turbo'})
    buffer = io.StringIO()
    data.to_csv(buffer, index=False)
    buffer.seek(0)
    
    agent = create_csv_agent(llm, buffer, verbose=True)
    st.write(agent.run(user_question))

# PDF functiondef 
def process_pdf_question(pdf):
    pdfReader = PdfReader(pdf)

    raw_text = ''
    for i, page in enumerate(pdfReader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=70,
        length_function=len,
    )
    pdfTexts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings(openai_api_key="",
                                  deployment="text-embedding-ada-002",
                                  client="azure")

    knowledge_base = FAISS.from_texts(pdfTexts, embeddings)
    # Your template
    template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. 
        Keep the answer as concise as possible. 
        Always say "thanks for asking!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    user_question = st.text_input("Ask a question from your image:")
    if user_question:
        llm = AzureOpenAI(
                temperature=0,
                openai_api_key="",
                deployment_name="gpt-35-turbo",
                model_name="gpt-35-turbo"
            )

        docs = knowledge_base.similarity_search(user_question, k=1)

        qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=knowledge_base.as_retriever(),
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )

        result = qa_chain({"query": user_question})
        answer = result["result"][:-10]
        st.write(answer)

# Image 
def process_image_question(uploaded_file):
    # Save the uploaded image to a temporary file
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Perform OCR on the temporary file
    reader = easyocr.Reader(["en"])  # language
    results = reader.readtext(temp_file_path)

    # Delete the temporary file
    os.remove(temp_file_path)

    # Extract the textt
    text = " "
    if results is not None:
        for result in results:
                # Check if the second element of the tuple is a string
            if isinstance(result[1], str):
                text += result[1] + " "

        text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=10,
                chunk_overlap=0,
                length_function=len)
        texts = text_splitter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key="",
                                        deployment="text-embedding-ada-002",
                                        client="azure")

        knowledge_base = FAISS.from_texts(texts, embeddings)

        # Template for question answering
        template = """Use the following pieces of context to answer the question at the end.
            If the question is not related to Image, say that it is not related to Image,
            don't try to make up an answer. Understand table values also. Always say "thanks for asking!" at the end of the answer.
            Image Context: {context}
            Question: {question}
            Helpful Answer:
            
            """
            
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

        # Show user input
        user_question = st.text_input("Ask a question from your image:")
        if user_question:
            llm = AzureOpenAI(temperature=0,
                              openai_api_key="",
                              deployment_name="gpt-35-turbo",
                              model_name="gpt-35-turbo")

            docs = knowledge_base.similarity_search(user_question, k=1)

            qa_chain = RetrievalQA.from_chain_type(llm,
                                                   retriever=knowledge_base.as_retriever(),
                                                   return_source_documents=True,
                                                   chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
            result = qa_chain({"query": user_question})
            answer = result["result"][:-10]

            st.write(answer)
# Database function

# ----------------------------------------------------------------------------------------------------------------
def main():
    st.header("Ask your CSV, PDF, Excel, and Image")

    options_sidebar = st.sidebar.radio("**Select file type:**", ["CSV", "PDF", "Excel", "Image"])

    if options_sidebar == "CSV":
        csv_file = st.sidebar.file_uploader("**Upload a CSV file**", type="csv")
        if csv_file:
            user_question = st.text_input("**Ask a question about your CSV Data:**")
            if user_question:
                process_csv_question(csv_file, user_question)

    elif options_sidebar == "PDF":
        pdf_file = st.sidebar.file_uploader("**Upload a PDF file**", type="pdf")
        if pdf_file:
            process_pdf_question(pdf_file)

    elif options_sidebar == "Image":
        image_file = st.sidebar.file_uploader("**Upload an Image file**", type=["jpeg", "jpg", "png"])
        if image_file:
            process_image_question(image_file)

    elif options_sidebar == "Excel":
        excel_file = st.sidebar.file_uploader("**Upload an Excel File**", type=["xlsx", "xls"])
        if excel_file:
            data = pd.read_excel(excel_file, engine="openpyxl")
            user_question = st.text_input("**Ask a question about your Excel Data:**")
            if user_question:
                process_Excel_question(data, user_question)

if __name__ == "__main__":
    main()