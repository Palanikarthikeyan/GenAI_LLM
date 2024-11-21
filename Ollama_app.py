import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

# prompt template

prompt = ChatPromptTemplate.from_messages(
    [ ("system","you are a helpful AI assistant,Please respond to the Q asked"),
     ("user","Question:{question}")
    ]
)
# streamlit frameword
st.title("Langchain Demo with Gemma2 model")
input_text = st.text_input("What question you have in mind?")


## Ollama model
llm = Ollama(model = "gemma:2b")
output = StrOutputParser()
chain = prompt|llm|output

if input_text:
    st.write(chain.invoke({"question":input_text}))





