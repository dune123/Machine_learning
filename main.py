embed_model = "text-embedding-ada-002"

import openai, langchain, pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
import streamlit as st


st.set_page_config(page_title='ASK QUESTION WITH PDF')
st.header('Ask your PDF')


pdf=st.file_uploader("Upload your pdf here",type=["pdf"])
text = ""
if pdf is not None:
    pdf_reader=PdfReader(pdf)

    for page in pdf_reader.pages:
        text+=page.extract_text()

#st.write(text)

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 2000,
    chunk_overlap  = 0,
    length_function = len,
)

pdf_text= text_splitter.create_documents([text])

#st.write(pdf_text)

#pinecone related setup
pinecone.init(
    api_key=st.secrets["PINECONE_API_KEY"],
    environment='gcp-starter'
)
#pinecoen related setup ends

index_name='testsearchbook'

embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

if index_name not in pinecone.list_indexes():
    print("Index does not exist: ", index_name)


book_docsearch = Pinecone.from_texts([t.page_content for t in pdf_text], embeddings, index_name = index_name)

#qa_chain_answering system

llm=OpenAI(temperature=0,openai_api_key=st.secrets['OPENAI_API_KEY'])

query=st.text_input("Enter your question")
if query:
    docs = book_docsearch.similarity_search(query)

    chain = load_qa_chain(llm, chain_type="stuff")
    answer=chain.run(input_documents=docs, question=query)
    st.write(answer)