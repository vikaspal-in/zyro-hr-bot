import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🤖")

# ---- Secrets (set these in Streamlit Cloud's "Secrets" settings) ----
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Optional: only needed if you also want this deployed app to log traces
if "LANGCHAIN_API_KEY" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

CORPUS_PATH = "data/"  # the PDFs will live in a "data" folder next to this file


@st.cache_resource(show_spinner="Setting up the HR knowledge base...")
def build_pipeline():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10},
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512)

    rag_prompt = ChatPromptTemplate.from_template(
        """You are an HR assistant for Zyro Dynamics. Answer the employee's
question using ONLY the context below from official HR policy documents.
If the answer isn't in the context, say you don't have that information.
Be clear, concise, and professional.

Context:
{context}

Question: {question}

Answer:"""
    )

    oos_prompt = ChatPromptTemplate.from_template(
        """Decide if the question below is about Zyro Dynamics HR policies
(leave, WFH, conduct, performance, compensation, IT/data security, POSH,
onboarding/separation, travel & expense, company profile, employee handbook).

Reply with only one word: IN_SCOPE or OUT_OF_SCOPE.

Question: {question}"""
    )

    return retriever, llm, rag_prompt, oos_prompt


retriever, llm, rag_prompt, oos_prompt = build_pipeline()

REFUSAL_MESSAGE = (
    "I can only answer HR-related questions from Zyro Dynamics policy "
    "documents. Please ask me about leave, WFH, compensation, conduct, "
    "or other HR policies."
)


def ask_bot(question: str):
    classifier_chain = oos_prompt | llm | StrOutputParser()
    verdict = classifier_chain.invoke({"question": question}).strip().upper()

    if "OUT_OF_SCOPE" in verdict:
        return REFUSAL_MESSAGE, []

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, docs


# ---------------- Chat UI ----------------
st.title("🤖 Zyro Dynamics HR Help Desk")
st.caption("Ask me about leave, WFH, compensation, conduct, and other HR policies.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask_bot(question)
            st.write(answer)
            if sources:
                with st.expander("📄 Sources"):
                    for i, doc in enumerate(sources, 1):
                        name = os.path.basename(doc.metadata.get("source", "unknown"))
                        page = doc.metadata.get("page", "?")
                        st.write(f"{i}. {name} (page {page})")

    st.session_state.messages.append({"role": "assistant", "content": answer})
