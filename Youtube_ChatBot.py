import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(page_title="YouTube ChatBot", page_icon="🎥", layout="centered")

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🤖 About the App")
    st.markdown("""
This application allows users to **interact with YouTube videos using AI**.  
By extracting transcripts, chunking content, embedding it, and using Gemini for reasoning — it delivers accurate, context-aware answers to your questions.

### 🔍 Key Features:
- YouTube transcript extraction  
- Contextual Q&A using Gemini AI  
- Embedding & vector search with FAISS  
- Clean user interface with chat input  
    """)
    st.markdown("---")
    
    st.markdown("## 👨‍💻 Developer Profile")
    st.markdown("""
**Name:** Tanjil Mahmud Emtu  
**Role:** AI & Software Developer  
**Expertise:**  
- Generative AI  
- LangChain & LLM apps  
- Chrome Extensions  
- NLP, Embeddings, Vector DBs  
- Python, JavaScript, Streamlit  
    """)
    st.markdown("""
🔗 **Connect with me:**  
[![GitHub](https://img.shields.io/badge/GitHub-000?logo=github&logoColor=white)](https://github.com/your-username)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/your-link)  
""", unsafe_allow_html=True)

    st.markdown("> *“I build AI tools that solve real problems — with clarity, speed, and focus.”*")

# --- Style ---
st.markdown("""
    <style>
    .big-title {
        font-size: 2.8em;
        font-weight: bold;
        text-align: center;
        color: #4A90E2;
        margin-bottom: 0.2em;
    }
    .sub-text {
        font-size: 1.2em;
        text-align: center;
        color: #6c757d;
        margin-bottom: 2em;
    }
    .info-box {
        background: linear-gradient(135deg, #e0f7fa, #fce4ec);
        padding: 15px;
        border-radius: 12px;
        margin-top: 15px;
        font-size: 1.05em;
        color: #333;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
    }
    .stTextInput > div > div > input {
        border: 2px solid #4A90E2;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.markdown('<div class="big-title">🎥 YouTube Video Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Ask questions directly from any YouTube video’s content</div>', unsafe_allow_html=True)

# --- Load Environment ---
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')

# --- Input Section ---
st.image("image.png", caption="🔗 Provide only the YouTube Video ID", use_container_width=True)
video_id = st.text_input("📺 Enter YouTube Video ID:", "")

transcript = ""
if video_id:
    with st.spinner("🔍 Fetching transcript..."):
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            transcripts = [chunk["text"] for chunk in transcript_list]
            transcript = " ".join(transcripts)
            st.success("✅ Transcript successfully retrieved.")
        except Exception as e:
            st.error(f"❌ No captions available for this video. Error: {str(e)}")
            transcript = ""

# --- Proceed if transcript available ---
if transcript:
    with st.spinner("⚙️ Processing transcript and building memory..."):
        spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = spliter.create_documents([transcript])

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            task="text-generation",
            api_key=os.getenv("GEMINI_API_KEY")
        )

        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.

            {context}
            Question: {question}
            """,
            input_variables=['context', 'question']
        )

        def format_docs(retrieved_docs):
            if not retrieved_docs:
                return "No relevant context found."
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser

    question = st.chat_input("💬 Ask a question about the video content:")
    if question:
        with st.spinner("🧠 Thinking..."):
            try:
                answer = main_chain.invoke(question)
                st.markdown("### 💡 Answer:")
                st.markdown(f"<div class='info-box'>{answer}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ An error occurred while processing your question. Error: {str(e)}")
