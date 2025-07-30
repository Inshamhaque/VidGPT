# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import videodb
from videodb import IndexType, SearchType
import os
from dotenv import load_dotenv
import boto3
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import uuid
import re
import logging
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Video Chat API", description="Backend API for video chat functionality")

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",'0.0.0.0'],  # Add your Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI models
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"]
)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"]
)

# Initialize VideoDB
conn = videodb.connect(api_key=os.environ["VIDEODB_API_KEY"])
coll = conn.get_collection()

# Initialize S3
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_REGION", "us-east-1")
)

# In-memory storage for video sessions
video_sessions = {}

# Pydantic models
class FileUploadResponse(BaseModel):
    file_url: str
    session_id: str
    message: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class TimestampResult(BaseModel):
    phrase: str
    start_time: float
    end_time: float

class ChatResponse(BaseModel):
    answer: str
    quotes: List[str]
    timestamps: List[TimestampResult]



class VideoSession:
    def __init__(self, video, video_url):
        self.video = video
        self.video_url = video_url
        self.transcript_text = None
        self.llama_index = None
        self.processing_status = {
            'transcript_generated': False,
            'index_created': False,
            'spoken_words_indexed': False,
            'processing_complete': False,
            'error': None
        }
    
    async def process_video_async(self):
        """Process video asynchronously during upload"""
        try:
            logger.info("🚀 Starting video processing...")
            
            # Step 1: Generate transcript
            await self._generate_transcript_async()
            
            # Step 2: Create searchable index
            await self._create_index_async()
            
            # Step 3: Index spoken words
            await self._index_spoken_words_async()
            
            self.processing_status['processing_complete'] = True
            logger.info("✅ Video processing complete! Chat is ready.")
            
        except Exception as e:
            error_msg = f"Video processing failed: {str(e)}"
            logger.error(error_msg)
            self.processing_status['error'] = error_msg
            self.processing_status['processing_complete'] = False
    
    async def _generate_transcript_async(self):
        """Generate transcript in background thread"""
        def generate_transcript():
            logger.info("📝 Generating transcript...")
            self.video.generate_transcript()
            self.transcript_text = self.video.get_transcript_text()
            return self.transcript_text
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            self.transcript_text = await loop.run_in_executor(executor, generate_transcript)
        
        self.processing_status['transcript_generated'] = True
        logger.info("✅ Transcript generated successfully")
    
    async def _create_index_async(self):
        """Create LlamaIndex in background thread"""
        def create_index():
            logger.info("🔍 Creating searchable index...")
            document = Document(text=self.transcript_text)
            splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            nodes = splitter.get_nodes_from_documents([document])
            return VectorStoreIndex(nodes), len(nodes)
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            self.llama_index, chunk_count = await loop.run_in_executor(executor, create_index)
        
        self.processing_status['index_created'] = True
        logger.info(f"✅ Index created with {chunk_count} chunks")
    
    async def _index_spoken_words_async(self):
        """Index spoken words in background thread"""
        def index_spoken_words():
            logger.info("🎤 Indexing spoken words...")
            self.video.index_spoken_words()
            return True
        
        try:
            # Run in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, index_spoken_words)
            
            self.processing_status['spoken_words_indexed'] = True
            logger.info("✅ Spoken words indexed successfully")
        except Exception as e:
            logger.warning(f"⚠️ Spoken words indexing failed: {e}")
            # Don't fail the entire process for this
            self.processing_status['spoken_words_indexed'] = False
    
    def query_video(self, user_query):
        """Query video - all processing already done!"""
        if not self.processing_status['processing_complete']:
            if self.processing_status['error']:
                raise Exception(f"Video processing failed: {self.processing_status['error']}")
            else:
                raise Exception("Video is still processing. Please wait a moment.")
        
        logger.info(f"⚡ Fast query (using cached data): {user_query}")
        
        query_engine = self.llama_index.as_query_engine(
            similarity_top_k=3,
            response_mode="tree_summarize"
        )
        
        enhanced_query = f"""
        Based on the video transcript, please answer this question: {user_query}
        
        In your response, please:
        1. Provide a clear, comprehensive answer
        2. Include exact quotes from the transcript that support your answer
        3. Format your response as:
        ANSWER: [your answer here]
        QUOTES: [exact quotes from transcript separated by | ]
        """
        
        response = query_engine.query(enhanced_query)
        return self.parse_response(str(response))
    
    def parse_response(self, response_text):
        """Parse AI response - instant since everything is cached"""
        try:
            if "ANSWER:" in response_text and "QUOTES:" in response_text:
                parts = response_text.split("QUOTES:")
                answer = parts[0].replace("ANSWER:", "").strip()
                quotes_text = parts[1].strip()
                quotes = [q.strip().strip('"').strip("'") for q in quotes_text.split("|") if q.strip()]
            else:
                answer = response_text
                quotes = re.findall(r'"([^"]+)"', response_text)
            
            # Get timestamps - super fast since spoken words already indexed
            timestamps = self.search_timestamps(quotes[:3]) if quotes else []
            
            return {
                "answer": answer,
                "quotes": quotes,
                "timestamps": timestamps
            }
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {
                "answer": response_text,
                "quotes": [],
                "timestamps": []
            }
    
    def search_timestamps(self, phrases):
        """Search timestamps - instant since already indexed"""
        if not self.processing_status['spoken_words_indexed']:
            logger.warning("Spoken words not indexed - skipping timestamps")
            return []
        
        timestamps = []
        for phrase in phrases:
            try:
                result = self.video.search(
                    query=phrase,
                    search_type=SearchType.semantic,
                    index_type=IndexType.spoken_word
                )
                
                if result.shots:
                    for shot in result.shots:
                        timestamps.append(TimestampResult(
                            phrase=phrase,
                            start_time=shot.start,
                            end_time=shot.end
                        ))
            except Exception as e:
                logger.error(f"Error searching phrase '{phrase}': {e}")
        
        return timestamps


# Pydantic model for URL upload
class URLUpload(BaseModel):
    url: str

@app.post("/upload-url")
async def upload_from_url(url_data: URLUpload):
    """Upload video from YouTube URL with background processing"""
    try:
        logger.info(f"📤 Uploading video from URL: {url_data.url}")
        
        # Upload to VideoDB
        video = coll.upload(url=url_data.url)
        
        # Create session
        session_id = str(uuid.uuid4())
        session = VideoSession(video, url_data.url)
        video_sessions[session_id] = session
        
        # Start background processing
        asyncio.create_task(session.process_video_async())
        
        logger.info(f"✅ Session created: {session_id}")
        logger.info("🔄 Background processing started...")
        
        return {
            "file_url": url_data.url,
            "session_id": session_id,
            "message": "Video uploaded successfully. Processing in background...",
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"❌ Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")

@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload file to S3 with background processing"""
    try:
        # Generate unique filename
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'mp4'
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        
        # Upload to S3 with public read permissions
        bucket_name = os.environ["S3_BUCKET_NAME"]
        s3_client.upload_fileobj(
            file.file,
            bucket_name,
            unique_filename,
            ExtraArgs={
                'ContentType': file.content_type,
                'ACL': 'public-read'
            }
        )
        
        # Generate S3 URL
        file_url = f"https://{bucket_name}.s3.{os.environ.get('AWS_REGION', 'us-east-1')}.amazonaws.com/{unique_filename}"
        
        logger.info(f"📤 Uploaded file to S3: {file_url}")
        
        # Upload to VideoDB
        video = coll.upload(url=file_url)
        
        # Create session and start processing
        session_id = str(uuid.uuid4())
        session = VideoSession(video, file_url)
        video_sessions[session_id] = session
        
        # Start background processing
        asyncio.create_task(session.process_video_async())
        
        logger.info(f"✅ Session created: {session_id}")
        logger.info("🔄 Background processing started...")
        
        return FileUploadResponse(
            file_url=file_url,
            session_id=session_id,
            message="File uploaded successfully. Processing in background..."
        )
        
    except Exception as e:
        logger.error(f"❌ Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

# New endpoint to check processing status
@app.get("/status/{session_id}")
async def get_processing_status(session_id: str):
    """Check if video processing is complete"""
    session = video_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    status = session.processing_status.copy()
    status.update({
        'session_id': session_id,
        'video_url': session.video_url,
        'ready_for_chat': status['processing_complete']
    })
    
    return status

@app.post("/chat", response_model=ChatResponse)
async def chat_with_video(chat_request: ChatRequest):
    """Process chat message and return AI response with timestamps"""
    try:
        session = video_sessions.get(chat_request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.info(f"Processing query for session {chat_request.session_id}: {chat_request.message}")
        
        # Process the query
        result = session.query_video(chat_request.message)
        
        return ChatResponse(
            answer=result["answer"],
            quotes=result["quotes"],
            timestamps=result["timestamps"]
        )
        
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "sessions": len(video_sessions),
        "videodb_connected": bool(conn),
        "s3_configured": bool(s3_client)
    }

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information"""
    session = video_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "video_url": session.video_url,
        "has_transcript": bool(session.transcript_text),
        "has_index": bool(session.llama_index)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)# main.py
