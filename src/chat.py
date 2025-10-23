"""
chat.py — chatbot Marvel (retrieval + mémoire) utilisant FAISS + Cohere + Groq
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
    print("🤖 Marvel Chat — propulsé par Groq + Cohere\n")
    print("Tape 'exit' pour quitter.\n")

    # Charger embeddings + base
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Config LLM (Groq)
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)

    # Mémoire conversationnelle
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Chaîne principale
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )

    # Boucle de chat
    while True:
        query = input("🧑‍💬 Toi : ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 À plus !")
            break

        result = chain.invoke({"question": query})
        print(f"🤖 Bot : {result['answer']}\n")


if __name__ == "__main__":
    main()
