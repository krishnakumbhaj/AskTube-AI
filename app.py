import streamlit as st
import time
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .ai-response {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    .ai-response h3 {
        color: #333;
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }
    
    .typing-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #333;
        font-family: 'Georgia', serif;
    }
    
    .status-success {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
    }
    
    .status-error {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
    
    .question-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid #e9ecef;
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.25);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

def typing_effect(text, placeholder, delay=0.03):
    """Create a typing effect for text display with enhanced styling"""
    displayed_text = ""
    for char in text:
        displayed_text += char
        placeholder.markdown(f"""
        <div class="ai-response">
            <h3>🤖 AI Assistant Response</h3>
            <div class="typing-text">{displayed_text}<span style="animation: blink 1s infinite;">▌</span></div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(delay)
    
    # Final display without cursor
    placeholder.markdown(f"""
    <div class="ai-response">
        <h3>🤖 AI Assistant Response</h3>
        <div class="typing-text">{displayed_text}</div>
    </div>
    """, unsafe_allow_html=True)

# Custom header
st.markdown("""
<div class="main-header">
    <h1>🎥 AskTube AI</h1>
    <p style="text-align: center; color: white; opacity: 0.9; font-size: 1.2rem; margin-top: 0.5rem;">
        Powered by Google Gemini AI
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced styling
st.sidebar.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)

# API Key input
api_key = st.sidebar.text_input("🔑 Google API Key", type="password", value="", help="Enter your Google API Key")

# Video ID input with enhanced styling
st.markdown('<div class="feature-card">', unsafe_allow_html=True)
st.markdown("### 📹 YouTube Video Setup")
video_id = st.text_input("YouTube Video ID", value="", help="Paste the YouTube video ID here", placeholder="e.g., dQw4w9WgXcQ")
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar controls for typing effect
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎭 Typing Effect Settings")
typing_speed = st.sidebar.slider("⚡ Typing Speed", min_value=0.01, max_value=0.1, value=0.03, step=0.01, help="Lower values = faster typing")
enable_typing = st.sidebar.checkbox("✨ Enable Typing Effect", value=True)

# Add some helpful information
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("• Use specific questions for better answers")
st.sidebar.markdown("• Try asking about key points or summaries")
st.sidebar.markdown("• The AI only uses video transcript content")

if api_key and video_id:
    with st.spinner("🔍 Fetching transcript..."):
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            transcript = " ".join([chunk['text'] for chunk in transcript_list])
            st.markdown('<div class="status-success">✅ Transcript fetched successfully! Ready to answer questions.</div>', unsafe_allow_html=True)
        except TranscriptsDisabled:
            st.markdown('<div class="status-error">❌ No captions available for this video.</div>', unsafe_allow_html=True)
            st.stop()
        except Exception as e:
            st.markdown(f'<div class="status-error">❌ Error fetching transcript: {e}</div>', unsafe_allow_html=True)
            st.stop()

    # Splitting the transcript
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.create_documents([transcript])

    # Embeddings and vector store
    with st.spinner("🧠 Creating vector store..."):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Model and prompt
    Model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.1
    )
    prompt = PromptTemplate(
        template="""You are a helpful assistant. 
        Answer ONLY from the context provided transcript context.
        If the context is not sufficient to answer the question, say "I don't know".
        {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
    )

    def format_docs(retrieved_docs):
        context_text = "\n".join([doc.page_content for doc in retrieved_docs])
        return context_text

    parallel_chain = RunnableParallel({
        'question': RunnablePassthrough(),
        'context': retriever | RunnableLambda(format_docs)
    })
    parser = StrOutputParser()
    final_prompt = parallel_chain | prompt | Model | parser

    # Question input with enhanced styling
    st.markdown('<div class="question-container">', unsafe_allow_html=True)
    st.markdown("### 💭 Ask Your Question")
    user_question = st.text_input("Your question", value="Can you summarize the video?", placeholder="What would you like to know about this video?")
    st.markdown('</div>', unsafe_allow_html=True)

    # Create two columns for buttons
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ask_button = st.button("🚀 Get Answer", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

    if ask_button and user_question:
        with st.spinner("🤖 Generating answer..."):
            response = final_prompt.invoke(user_question)
        
        # Create placeholder for typing effect
        response_placeholder = st.empty()
        
        if enable_typing:
            # Display response with typing effect
            typing_effect(response, response_placeholder, typing_speed)
        else:
            # Display response normally with styling
            response_placeholder.markdown(f"""
            <div class="ai-response">
                <h3>🤖 AI Assistant Response</h3>
                <div class="typing-text">{response}</div>
            </div>
            """, unsafe_allow_html=True)
            
    if clear_button:
        st.rerun()

else:
    st.markdown("""
    <div class="info-box">
        <h3>🚀 Get Started</h3>
        <p>Please enter your Google API Key and YouTube Video ID to begin exploring!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add some instructions
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### 📖 How to Use")
    st.markdown("1. **Get API Key**: Obtain your Google API Key from Google Cloud Console")
    st.markdown("2. **Find Video ID**: Copy the YouTube video ID from the URL (e.g., `dQw4w9WgXcQ` from `youtube.com/watch?v=dQw4w9WgXcQ`)")
    st.markdown("3. **Ask Questions**: Type your questions about the video content")
    st.markdown("4. **Get AI Answers**: Watch the AI respond with information from the video transcript")
    st.markdown('</div>', unsafe_allow_html=True)