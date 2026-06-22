import streamlit as st
import re
import os
from dotenv import load_dotenv

load_dotenv()

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .chat-message {
        padding: 1.8rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #1e2a3a;
        border: 2px solid #4a90e2;
    }
    .assistant-message {
        background-color: #2a2a2a;
        border: 2px solid #555;
    }
    .chat-message strong {
        font-size: 1.3rem;
        color: #ffffff;
        margin-bottom: 1rem;
        display: block;
        font-weight: 700;
    }
    .chat-message {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)

# Show loading messagec
with st.spinner("Loading dependencies... This may take a moment on first run."):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from langchain_groq import ChatGroq

class YouTubeRAG:
    def __init__(self, model_name="llama-3.3-70b-versatile", persist_directory="./youtube_chroma_db"):
        """Initialize the RAG system"""
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.retriever = None
    
    def _init_embeddings(self):
        """Initialize embeddings model (lazy loading)"""
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
    
    def _init_llm(self):
        """Initialize LLM (lazy loading)"""
        if self.llm is None:
            self.llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=self.model_name,
                temperature=0.7,
                max_tokens=512,
            )
        
    def extract_video_id(self, url):
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=)([^&]+)',
            r'(?:youtu\.be\/)([^?]+)',
            r'(?:youtube\.com\/embed\/)([^?]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return url  # Assume it's already a video ID
        
    def get_youtube_transcript(self, video_id):
        """Fetch transcript from YouTube video using yt-dlp"""
        try:
            import yt_dlp
            import urllib.request
            import json
            
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            ydl_opts = {
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en'],
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                video_title = info.get('title', 'Unknown')
                
                # Try to get subtitles
                subtitles = None
                
                if 'subtitles' in info and 'en' in info.get('subtitles', {}):
                    subtitles = info['subtitles']['en']
                elif 'automatic_captions' in info and 'en' in info.get('automatic_captions', {}):
                    subtitles = info['automatic_captions']['en']
                else:
                    return None, None
                
                # Find json3 format
                subtitle_url = None
                for sub in subtitles:
                    if sub.get('ext') == 'json3':
                        subtitle_url = sub.get('url')
                        break
                
                if not subtitle_url:
                    return None, None
                
                # Fetch the subtitle content
                with urllib.request.urlopen(subtitle_url) as response:
                    subtitle_data = json.loads(response.read().decode('utf-8'))
                
                # Extract text
                transcript_text = ""
                if 'events' in subtitle_data:
                    for event in subtitle_data['events']:
                        if 'segs' in event:
                            for seg in event['segs']:
                                if 'utf8' in seg:
                                    transcript_text += seg['utf8']
                
                # Clean up
                transcript_text = re.sub(r'\s+', ' ', transcript_text).strip()
                
                return transcript_text, video_title
                
        except Exception as e:
            return None, str(e)
    
    def create_vectorstore(self, transcript_text, video_id):
        """Create vector store from transcript"""
        # Initialize embeddings if not already done
        self._init_embeddings()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_text(transcript_text)
        
        documents = [
            Document(
                page_content=chunk,
                metadata={"source": f"youtube_{video_id}", "chunk": i}
            )
            for i, chunk in enumerate(chunks)
        ]
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=f"{self.persist_directory}_{video_id}"
        )
        
        return len(chunks)
    
    def setup_retriever(self):
        """Setup the retriever"""
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
    def process_video(self, video_url):
        """Process a YouTube video and setup for Q&A"""
        video_id = self.extract_video_id(video_url)
        transcript, video_title_or_error = self.get_youtube_transcript(video_id)
        
        if transcript is None:
            return False, None, None, video_title_or_error
        
        num_chunks = self.create_vectorstore(transcript, video_id)
        self.setup_retriever()
        
        return True, video_title_or_error, num_chunks, None
    
    def ask_question(self, question):
        """Ask a question about the video"""
        if self.retriever is None:
            return "Please process a video first!"
        
        # Initialize LLM if not already done
        self._init_llm()
        
        relevant_docs = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""Use the following context from a YouTube video transcript to answer the question.
        If you don't know the answer based on the context, say so.

        Context: {context}

        Question: {question}

        Answer: """
                
        answer = self.llm.invoke(prompt)
        return answer.content

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = None
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_title' not in st.session_state:
    st.session_state.video_title = None

# Main UI
st.title("🎥 YouTube RAG Chatbot")
st.markdown("Ask questions about any YouTube video!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("🔑 Using Groq API (`llama-3.3-70b-versatile`)")

    st.divider()

    st.header("📹 Video Input")
    
    video_url = st.text_input(
        "YouTube URL or Video ID",
        placeholder="https://www.youtube.com/watch?v=...",
        key="video_url_input"
    )
    
    if st.button("🔄 Process Video", type="primary", use_container_width=True):
        if video_url:
            with st.spinner("Processing video... This may take a minute on first run."):
                # Initialize RAG
                if st.session_state.rag is None:
                    st.session_state.rag = YouTubeRAG()
                
                # Process video
                success, video_title, num_chunks, error = st.session_state.rag.process_video(video_url)
                
                if success:
                    st.session_state.video_processed = True
                    st.session_state.video_title = video_title
                    st.session_state.chat_history = []
                    st.success(f"✅ Video processed successfully!")
                    st.info(f"📝 Title: {video_title}")
                    st.info(f"📊 Created {num_chunks} text chunks")
                else:
                    st.error(f"❌ Failed to process video: {error or 'Unknown error'}")
        else:
            st.warning("⚠️ Please enter a YouTube URL")
    
    if st.session_state.video_processed:
        st.divider()
        st.success("✅ Video Ready!")
        if st.session_state.video_title:
            st.markdown(f"**Current Video:**  \n{st.session_state.video_title}")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Main chat area
if st.session_state.video_processed:
    st.markdown("### 💬 Chat with the Video")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🧑 You:</strong><br>
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    question = st.chat_input("Ask a question about the video...")
    
    if question:
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        
        # Get answer
        with st.spinner("Thinking..."):
            answer = st.session_state.rag.ask_question(question)
        
        # Add assistant message to chat
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })
        
        st.rerun()

else:
    # Welcome screen
    st.markdown("""
        ### 👋 Welcome!

        To get started:
        1. Paste a YouTube URL or video ID in the sidebar
        2. Click "Process Video"
        3. Start asking questions!
        
        #### Example URLs:
        - `https://www.youtube.com/watch?v=GbNjHKRzDiQ`
        - `GbNjHKRzDiQ` (just the video ID)
        
        #### Example Questions:
        - What is this video about?
        - Summarize the main points
        - What are the key takeaways?
    """)