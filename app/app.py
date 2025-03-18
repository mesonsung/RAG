import asyncio
import nest_asyncio
import streamlit as st
from langchain.schema.runnable import RunnableSequence
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM as Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import RetrievalQA
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Ensure proper event loop handling
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

nest_asyncio.apply()

# Streamlit UI
st.title("📄 RAG 系統與 DeepSeek R1 和 Ollama")

uploaded_file = st.file_uploader("在這裡上傳你的 PDF 文件", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings()
    text_splitter = SemanticChunker(embeddings=embeddings)
    documents = text_splitter.split_documents(docs)

    embedder = HuggingFaceEmbeddings()
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = Ollama(model="deepseek-r1:8b")

    prompt = """
    使用以下上下文回答問題。
    上下文: {context}
    問題: {question}
    答案:"""

    QA_PROMPT = PromptTemplate.from_template(prompt)

    # Replace LLMChain with RunnableSequence
    llm_chain = QA_PROMPT | llm
    combine_documents_chain = create_stuff_documents_chain(
        llm=llm_chain, 
        prompt=QA_PROMPT, 
        document_variable_name="context"
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    user_input = st.text_input("詢問你的文件問題：")

    if user_input:
        response = qa.invoke({"query": user_input})
        st.write("**回應：**")
        st.write(response["result"])  # Display the main result
        st.write("**來源文件：**")
        for doc in response["source_documents"]:
            st.write(doc.page_content)  # Display the content of the source documents