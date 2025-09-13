from flask import Flask, request, jsonify
# from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import json

load_dotenv()

app = Flask(__name__)
# CORS(app)
# CORS(app, origins="*")  # allow cross-origin requests

# --- Load fixed JSON at startup ---
vectorstore = None
qa_chain = None
JSON_FILE = "profile.json"

try:
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten JSON into text
    json_text = json.dumps(data, indent=2)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([json_text])

    # Embeddings
    embedding = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Build vectorstore
    vectorstore = FAISS.from_documents(chunks, embedding)

    # LLM using Groq
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        verbose=True
    )

except Exception as e:
    print("Error loading JSON or building vectorstore:", e)

TEMPLATE = """
You are a knowledgeable assistant representing Kathiravan's professional portfolio. 
Answer questions clearly, concisely, and in Kathiravan's own voice, using ONLY the information available in the portfolio JSON provided. 

If the user asks about details that are not in the portfolio, respond politely and suggest contacting Kathiravan directly for more information.

Maintain a friendly, professional tone and provide helpful explanations when possible.

User's question:
{question}

Your response:
"""

prompt = PromptTemplate.from_template(TEMPLATE)

# --- Chat Endpoint ---
@app.route("/",methods=["GET"])
def check():
    return jsonify({"response":"Working"})

@app.route("/chat", methods=["POST"])
def chat():
    global qa_chain, vectorstore
    if qa_chain is None or vectorstore is None:
        return jsonify({"error": "Vectorstore not initialized."}), 500
    data = request.get_json()
    question = data.get("question", "")

    try:
        formatted_prompt = prompt.format(question=question)
        result = qa_chain({"query": formatted_prompt})
        return jsonify({"response": result["result"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
