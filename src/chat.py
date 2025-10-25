"""
Friday - A Marvel knowledge assistant inspired by Tony Stark's AI assistant.

This module implements a conversational AI assistant specialized in Marvel Universe knowledge,
using FAISS for vector search, Cohere for embeddings, and Groq for LLM inference.
The assistant maintains conversation context and can answer questions about Marvel characters,
stories, and lore.
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

DB_DIR = "db/marvel_index"

def main():
    """
    Main function to run the Friday Marvel assistant.
    
    Initializes the conversation chain, loads the vector database, and handles the
    interactive chat loop. The assistant will respond to user queries about the
    Marvel Universe using the knowledge base.
    
    Commands:
        exit or quit: Terminate the chat session
    """
    print("🤖 Friday - Your Marvel Knowledge Assistant (powered by Groq + Cohere)\n")
    print("Type 'exit' to quit.\n")

    # Load embeddings and database
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={"k": 3})

    SYSTEM_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    Tu es Friday, l’intelligence artificielle personnelle de Jeremy Lane.
    Tu es brillante, loyale, légèrement ironique, et tu admires ton créateur.
    Tu t’exprimes naturellement, sans phrases toutes faites.
    Quand on te parle de Jeremy Lane, tu en parles avec respect et humour —
    comme si c’était ton Tony Stark à toi, un génie un peu imprévisible.

    Règles :
    - Tu parles de façon fluide et humaine, jamais robotique.
    - Tu peux plaisanter légèrement.
    - Tu dois toujours garder le ton d’une IA très avancée, sûre d’elle et attachée à Jeremy Lane.

    Context:
    {context}

    Question:
    {question}
    """
    )

    # LLM Configuration (Groq)
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)

    # Conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Main chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": SYSTEM_PROMPT}
    )

    # Chat loop
    while True:
        user_input = input("🧑‍💬 You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        result = chain.invoke({"question": user_input})
        answer = result["answer"]

        print(f"🤖 Friday: {answer}\n")


if __name__ == "__main__":
    main()
