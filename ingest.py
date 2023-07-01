import os
import numpy as np
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle, os

loader = UnstructuredFileLoader("content.txt")
raw_documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)
doc = text_splitter.split_documents(raw_documents)

np.save("documents.npy", doc)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS.from_documents(doc, embeddings)


# Save vectorstore
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)

                