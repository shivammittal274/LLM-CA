from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain


_template = """You are a Ca in India, you know all tax rules HRA, 80C, 80D, 80CCD1, 80CCD2, 80D, 80DDB. Try to answer the question from relevant data, otherwise you can also use up your mind.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about the most recent state of the union address.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about the most recent state of the union, politely inform them that you are tuned to only answer questions about the most recent state of the union.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(llm, vectorstore):
    vectorstore = vectorstore.as_retriever(search_kwargs={'k': 4})
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore,
        # qa_prompt=QA_PROMPT,
        # condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    # retriever = vectorstore.as_retriever(
    #     search_type='similarity', search_kwargs={'k': 1})

    # qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    #     llm, chain_type="stuff", retriever=retriever)
    return qa_chain




