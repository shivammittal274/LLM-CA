from os import environ
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.document_loaders import PyPDFLoader
import pickle
import gptcache
import time
import numpy as np
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.embedding import OpenAI as openai_embedding
import langchain
from langchain.llms import OpenAI
from gptcache.adapter.langchain_models import LangChainLLMs
import langchain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from cosine_sim_eval import CosineSimEvaluation
from langchain.memory import ConversationBufferMemory
from query_data import get_chain
from gptcache import cache
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI


# loader = PyPDFLoader("/home/ubuntu/shivam/pdf-summ/chapter1.pdf")
# pages = loader.load_and_split()
# print (f'You have {len(pages)} page(s) in your data')

# # We need to split long documents into smaller chunks to stay under the LLM token limit.
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(pages)
# print (f'Now you have {len(texts)} documents')
documents = np.load("documents.npy", allow_pickle=True)
embeddings = OpenAIEmbeddings()
# vectorstore = Milvus.from_documents(
#     texts,
#     embeddings,
#     connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
# )
vectorstore = FAISS.from_documents(documents, embeddings)
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
print (f'Milvus store created.')


#retriever = vector_db.as_retriever(search_type='similarity', search_kwargs={'k':1})
#chain = RetrievalQAWithSourcesChain.from_chain_type(llm_openai, chain_type="stuff", retriever=retriever)

def get_content_func(data, **_):
    return data.get("prompt").split("Question")[-1]

print("Cache loading.....")
openai_emb = openai_embedding()
# onnx = Onnx()

data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=openai_emb.dimension))
cache.init(
    pre_embedding_func=get_content_func,
    embedding_func=openai_emb.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=CosineSimEvaluation(),
    )
cache.set_openai_key()
# langchain.llm_cache = GPTCache(init_gptcache_map)
llm=ChatOpenAI(model='gpt-4', temperature=0)
chain = get_chain(llm, vectorstore=vectorstore)

questions = [
    "what's democracy",
    "what is democracy ",
    "Explain democracy",
    # "how democracy is related to elections",
    # "What are the different features of democracy",
    # "What are elections?",
    # "why is government needed",
    # "what is your name",
    "Is India a democracy",
    "What is food",
    "why are elections held",
    "summary on chapter 1",
    "give TL;DR summary of chapter 1",
    "How is democracy related to independence?"
]


def llm_qa(question, chat_history):
    # question += 'Instructions: No need to answer from text necessarily, If question is related to maths, answer using logic.'
    start_time = time.time()
    docs = (vectorstore.similarity_search(question))
    # response = chain({"question": question, "chat_history": []})
    
    mapped_qa = [[{"role": "user", "content": question}, {"role": "assistant", "content": answer}] for question, answer in chat_history]

    # Flatten the list of lists
    mapped_qa = [item for sublist in mapped_qa for item in sublist]
        
    import openai

    openai.api_key = "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"
    # question = "What deducations I can claim in India?"
    response = (openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
            {"role": "system", "content": """You are a CA having good knowledge of Indian Income tax rules. I would also try to share important information in context you may require
            while answering the question in Context. You can use that information while answering.
            Example: Context: User's Question: 
            
            And Lets say, if user asks something but you require more information from user, you can ask the user. 
            Example User ask can you compute my HRA component for me, ask whats your basic salary, HRA components etc whichever required to you """},
            *mapped_qa,
            {"role": "user", f"content": f"Context: {docs} \nQuestion: {question}"}
    ]
    ))
    print(response)
    
    # response = chain({"question": question}, return_only_outputs=True)
    # response = chain({"question": question}, return_only_outputs=True)
    time_taken = time.time() - start_time
    print("Time consuming: {:.2f}s".format(time_taken))
    return response['choices'][0]['message']['content'], "Time taken: {:.2f}s".format(time_taken)
    # print(f'Answer: {response}\n')
    return response['answer'], "Time taken: {:.2f}s".format(time_taken) 

if __name__ == '__main__':
    for question in questions:
        print(llm_qa(question=question))