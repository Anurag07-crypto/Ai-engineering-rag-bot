# ---------------------------------------------------------------
from typing import List
import streamlit as st
from pathlib import Path
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from embedding_manager import Embedding_manager
from vector_db import VectorDB
from rag import RAG
from pydantic import BaseModel, Field
# ---------------------------------------------------------------
# query = input("enter your query: ")
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["SARVAM_API_KEY"] = os.getenv("SARVAM_API_KEY")
# ---------------------------------------------------------------
def text_splitting(document, chunk_size:int=1000, chunk_overlap:int=200):
    
    text_splits = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function = len,
        separators=["\n\n","\n",""]
    )
    
    chunks = text_splits.split_documents(documents=document)
    print(f"{len(document)} document split into {len(chunks)} chunks")
    return chunks
# ---------------------------------------------------------------
@st.cache_resource
def build_vector_db():
    embedding_manager = Embedding_manager()
    vector_store = VectorDB()

    data = DirectoryLoader(
    path=Path("C:/Users/Lenovo/Desktop/Next Plan/data"),
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True
    )
    datas = data.load()


    chunk = text_splitting(datas)
    texts = [docs.page_content for docs in chunk]

    embeddings = embedding_manager.generate(texts)
    vector_store.add_docs(chunk, embeddings)
    return embedding_manager,vector_store

embedding_manager, vector_store = build_vector_db()
# ---------------------------------------------------------------

rag = RAG(vector_store, embedding_manager)

# helper to escape curly braces so ChatPromptTemplate doesn't treat them as variables

def escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")

# ---------------------------------------------------------------
llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    max_tokens=1024,
    temperature=0.0
)

# Structured output for better parsing
class StructuredOutput(BaseModel):
    definition: str = Field(
        description="Short explanation of the concept"
    )

    pros: List[str] = Field(
        description="Exactly 3 short advantages as separate list items"
    )

    cons: List[str] = Field(
        description="Exactly 3 short disadvantages as separate list items"
    )

    use_case: str = Field(
        description="Short explanation of where this is used"
    )

str_opt = llm.with_structured_output(StructuredOutput,
                                     method="function_calling")

st.title("Learn A to Z AI Engineering in this RAG BOT")
st.subheader("Ask any question releated to it")
col_1, col_2 = st.columns(2)

with col_1:
    st.write("Input")
    with st.form("Query Form"):
        query = st.text_input("Enter your query",
                              placeholder="What is pytorch")
        submit = st.form_submit_button("Submit")
    

with col_2:
    st.write("Output")
    if submit and query.strip():
        rag_response = rag.retrieve(query=query, threshold=0.5)

        if rag_response==None or rag_response==[] or rag_response=="":
            st.warning("No relevent Information Found")
            st.info("Try Rephrasing your Question")
            st.stop()
        # st.write("RAG RESULT:", rag_response)
        escaped_context = escape_braces(str(rag_response))
        system_prompt = f"""
You are a RAG assistant providing educational content.

MANDATORY RULES:
1. Answer EXCLUSIVELY from the context provided below
2. If information is not in context, set all fields to indicate "Not available in data"
3. Never use external knowledge or training data
4. Be concise and educational

CONTEXT:
{escaped_context}
"""
        prompt = ChatPromptTemplate.from_messages([
                ("system",system_prompt),
                ("human","{query}")
            ])
        chain = prompt | str_opt
        try:
            response = chain.invoke({"query": query}) 
            
            pros_text = "\n".join([f"- {p}" for p in response.pros])
            cons_text = "\n".join([f"- {c}" for c in response.cons])
            
            response_text = f"""
    ### Definition
    {response.definition}

    ### Pros
    {response.pros}

    ### Cons
    {response.cons}

    ### Concept of Use
    {response.use_case}
    """
            st.markdown(response_text)
        except Exception as e:
            st.error(f"Error generating response {str(e)}")