import os
import getpass

# language Model packages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader


# Test
prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
model = ChatOpenAI()

# attaching stop token to the prompt
chain = prompt | model.bind(stop=["\n"])
chain = prompt | model

chain.invoke({"foo": "bears"})

# NOTE Similarity Search#
# RAG SETUP
# Load and split PDF into pages
loader = PyPDFLoader(r"C:\Users\behna\Desktop\proposals\Awarded\DOI.pdf")
documents = loader.load_and_split()  # This should give you the split documents directly

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# Generate embeddings and create a FAISS vector store
embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
db = FAISS.from_documents(docs, embeddings)

query = "What is the Contract Vehicle for this award?"
docs = db.similarity_search(query)

print(docs[0].page_content)


# Check the core of the search
docs_and_scores = db.similarity_search_with_score(query)
docs_and_scores[0]