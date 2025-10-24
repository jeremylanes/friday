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
        verbose=False
    )

    # Chat loop
    while True:
        query = input("🧑‍💬 You: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        result = chain.invoke({"question": query})
        print(f"🤖 Friday: {result['answer']}\n")


if __name__ == "__main__":
    main()
