### PyPDFDirectoryLoader
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
## load the Groq API Key
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
gk=os.getenv('GROQ_API_KEY')

#embeddings = OllamaEmbeddings(model="gemma:2b")
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

llm = ChatGroq(model='Llama3-8b-8192',groq_api_key=gk)

prompt=ChatPromptTemplate.from_template(
    '''
    Answer the question based on the provided context only.
    provide the most accurate respone based on question
    <context>
    {context}
    <context>
    Question:{input}''')

st.title('RAG Document QA with Groq anad Lama3')
user_prompt = st.text_input('Enter your query from the research paper')

# This is Embedding - block
if st.button("Document Embedding"):
    f1() # below code
    st.write("Vector DB is ready")

'''

loader = PyPDFDirectoryLoader("research_papers")
docs = loader.load()
text=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
final_docs = text.split_documents(docs[:50])
FAISS.from_documents(final_docs,embedding)

'''
'''
doc_chain = create_stuff_documents_chain(llm.prompt)
retriver = vectors.as_retrivever()
retrieval_chain = create_retrieval_chain(retriver,doc_chain)
response = retrieval_chain.invoke({'input':user_prompt})
response['answer']
'''




