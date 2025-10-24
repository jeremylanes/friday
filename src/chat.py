"""
chat.py - Marvel chatbot (retrieval + memory) using FAISS + Cohere + Groq
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
    print("🤖 Marvel Chat — powered by Groq + Cohere\n")
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
        query = input("🧑‍💬 Toi : ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        result = chain.invoke({"question": query})
        print(f"🤖 Friday: {result['answer']}\n")


if __name__ == "__main__":
    main()
