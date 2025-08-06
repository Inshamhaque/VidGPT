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
from tempfile import NamedTemporaryFile
from openai import OpenAI as OpenAIClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Chat API", description="Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",'0.0.0.0'],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Settings.llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"]
)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"]
)

conn = videodb.connect(api_key=os.environ["VIDEODB_API_KEY"])
coll = conn.get_collection()

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_REGION", "us-east-1")
)

video_sessions = {}

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

class TranscriptRequest(BaseModel):
    session_id : str



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

        file_url = f"https://{bucket_name}.s3.{os.environ.get('AWS_REGION', 'us-east-1')}.amazonaws.com/{unique_filename}"

        logger.info(f"📤 Uploaded file to S3: {file_url}")

        video = coll.upload(url=file_url)

        session_id = str(uuid.uuid4())
        session = VideoSession(video, file_url)
        video_sessions[session_id] = session

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

@app.get("/transcript/{session_id}")
async def get_transcript(session_id):
    """Get session transcript"""
    session = video_sessions.get(session_id)
    if(not session):
        raise HTTPException(status_code=404,detail = "Session not found")
    if(not session.transcript_text):
        raise HTTPException(status_code=404,detail = "Session not indexed yet")
    return{
        "session_id":session_id,
        "transcript":session.transcript_text
    }


@app.post("/chapter/{session_id}")
async def get_chapters(session_id: str):
    """Get chapter-wise summary using VideoDB's spoken word indexing for accurate timestamps"""
    session = video_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.processing_status['processing_complete']:
        raise HTTPException(status_code=400, detail="Video processing not complete yet")

    if not session.processing_status['spoken_words_indexed']:
        return await get_chapters_fallback(session_id, session)

    try:
        topics = await identify_chapter_topics(session.transcript_text)
        
        chapters_with_timestamps = await find_precise_timestamps(session, topics)
        final_chapters = await generate_chapter_summaries(session, chapters_with_timestamps)
        
        return {
            "session_id": session_id,
            "chapters": final_chapters,
            "method": "videodb_enhanced"
        }

    except Exception as e:
        logger.error(f"Enhanced chapter generation failed: {e}")
        return await get_chapters_fallback(session_id, session)


async def identify_chapter_topics(transcript_text: str):
    """Use LLM to identify key topics and their keywords for searching"""
    from openai import OpenAI
    import json
    
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    prompt = f"""
Analyze this video transcript and identify 4-8 main topics/themes that would make good chapter divisions.

For each topic, provide:
1. A descriptive title
2. 3-5 key phrases or keywords that exactly match the transcript as we have to search through the index of spoken words
3. A brief description

Format as JSON:
[
  {{
    "title": "Introduction to Topic X",
    "keywords": ["keyword1", "specific phrase", "technical term"],
    "description": "Brief description of what this section covers"
  }},
  ...
]

Transcript:
\"\"\"
{transcript_text[:4000]}...
\"\"\"
"""
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert at analyzing video content and creating chapter structures."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    raw_output = response.choices[0].message.content.strip()
    if raw_output.startswith("```"):
        raw_output = re.sub(r"^```(?:json)?", "", raw_output)
        raw_output = raw_output.rstrip("```").strip()
    
    return json.loads(raw_output)


async def find_precise_timestamps(session: VideoSession, topics: List[dict]):
    """Use VideoDB's spoken word search to find precise timestamps for each topic"""
    chapters_with_timestamps = []
    
    for i, topic in enumerate(topics):
        best_timestamp = None
        best_score = 0

        for keyword in topic["keywords"]:
            try:
                # video db semantic searching
                result = session.video.search(
                    query=keyword,
                    search_type=SearchType.semantic,
                    index_type=IndexType.spoken_word
                )
                
                if result.shots and len(result.shots) > 0:
                    # first occurrence
                    shot = result.shots[0]
                    
                    if best_timestamp is None or shot.start < best_timestamp:
                        best_timestamp = shot.start
                        best_score = shot.search_score if hasattr(shot, 'search_score') else 1.0
                        
            except Exception as e:
                logger.warning(f"Failed to search for keyword '{keyword}': {e}")
                continue

        if best_timestamp is None:
            estimated_duration = session.video.length if hasattr(session.video, 'length') else 1800  # 30 min default
            best_timestamp = (i / len(topics)) * estimated_duration
        
        chapters_with_timestamps.append({
            "title": topic["title"],
            "description": topic["description"],
            "start_time": best_timestamp,
            "keywords": topic["keywords"]
        })
    
    chapters_with_timestamps.sort(key=lambda x: x["start_time"])
    
    return chapters_with_timestamps


async def generate_chapter_summaries(session: VideoSession, chapters_with_timestamps: List[dict]):
    """Generate final chapter summaries with precise timestamps"""
    final_chapters = []
    
    for i, chapter in enumerate(chapters_with_timestamps):
        start_time = chapter["start_time"]
        
        # Determine end time (start of next chapter or end of video)
        if i < len(chapters_with_timestamps) - 1:
            end_time = chapters_with_timestamps[i + 1]["start_time"]
        else:
            # Last chapter - estimate end time or use video length
            end_time = start_time + 300  # Default 5 minutes for last chapter
            if hasattr(session.video, 'length'):
                end_time = min(end_time, session.video.length)
        
        # Format timestamps
        start_formatted = seconds_to_mmss(start_time)
        
        # Get content for this time range using VideoDB search
        chapter_content = await get_chapter_content(session, chapter["keywords"], start_time, end_time)
        
        final_chapters.append({
            "title": chapter["title"],
            "start_time": start_formatted,
            "start_seconds": start_time,
            "description": chapter_content if chapter_content else chapter["description"],
            "duration": int(end_time - start_time)
        })
    
    return final_chapters


async def get_chapter_content(session: VideoSession, keywords: List[str], start_time: float, end_time: float):
    """Extract relevant content for a chapter using VideoDB search"""
    try:
        # Search for content in this time range
        relevant_content = []
        
        for keyword in keywords[:2]:  # Limit to top 2 keywords to avoid too many searches
            result = session.video.search(
                query=keyword,
                search_type=SearchType.semantic,
                index_type=IndexType.spoken_word
            )
            
            if result.shots:
                for shot in result.shots:
                    # Only include shots within our chapter time range
                    if start_time <= shot.start <= end_time:
                        relevant_content.append(shot.text if hasattr(shot, 'text') else keyword)
        
        if relevant_content:
            # Combine and summarize the content
            combined_content = " ".join(relevant_content[:3])  # Limit length
            return f"Covers {', '.join(keywords[:3])} and related topics. {combined_content[:100]}..."
        
        return None
        
    except Exception as e:
        logger.warning(f"Failed to get chapter content: {e}")
        return None


async def get_chapters_fallback(session_id: str, session: VideoSession):
    """Fallback method using basic LLM analysis (original implementation)"""
    from openai import OpenAI
    import json
    
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    prompt = f"""
You are given the full transcript of a video below. Your task is to divide the video into 4–8 coherent chapters.

For each chapter, provide:
- Title
- Approximate start time (in mm:ss format, estimate if exact timing not available)
- 1-sentence summary of the chapter

Format your response as a JSON list like this:
[
  {{
    "title": "Chapter Title",
    "start_time": "mm:ss",
    "description": "Short description of what is covered in this chapter"
  }},
  ...
]

Transcript:
\"\"\"
{session.transcript_text}
\"\"\"
    """.strip()
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes videos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        if raw_output.startswith("```"):
            raw_output = re.sub(r"^```(?:json)?", "", raw_output)
            raw_output = raw_output.rstrip("```").strip()
        
        chapters = json.loads(raw_output)
        
        return {
            "session_id": session_id,
            "chapters": chapters,
            "method": "llm_fallback"
        }
        
    except Exception as e:
        logger.error(f"Fallback chapter generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate chapters")


def seconds_to_mmss(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


# Additional endpoint to get chapters with video clips
@app.post("/chapter-clips/{session_id}")
async def get_chapters_with_clips(session_id: str):
    """Get chapters and generate video clips for each chapter"""
    session = video_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.processing_status['processing_complete']:
        raise HTTPException(status_code=400, detail="Video processing not complete yet")

    try:
        # Get enhanced chapters first
        chapters_response = await get_chapters(session_id)
        chapters = chapters_response["chapters"]
        
        # Generate clips for each chapter (if VideoDB supports it)
        chapters_with_clips = []
        
        for chapter in chapters:
            try:
                if "start_seconds" in chapter:
                    start_time = chapter["start_seconds"]
                    duration = chapter.get("duration", 60)  # Default 1 minute
                    
                    # Generate clip (if VideoDB supports timeline/clip generation)
                    # clip_url = session.video.generate_clip(start=start_time, duration=duration)
                    
                    chapter_with_clip = chapter.copy()
                    # chapter_with_clip["clip_url"] = clip_url
                    chapter_with_clip["timestamp_url"] = f"{session.video_url}#t={int(start_time)}"
                    
                    chapters_with_clips.append(chapter_with_clip)
                else:
                    chapters_with_clips.append(chapter)
                    
            except Exception as e:
                logger.warning(f"Failed to generate clip for chapter: {e}")
                chapters_with_clips.append(chapter)
        
        return {
            "session_id": session_id,
            "chapters": chapters_with_clips,
            "method": chapters_response.get("method", "enhanced")
        }
        
    except Exception as e:
        logger.error(f"Chapter clips generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate chapter clips")

@app.post("/whisper")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio using Whisper (OpenAI)"""
    try:
        # Create a direct OpenAI client for audio transcription
        openai_client = OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])
        
        # Save audio file temporarily
        with NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Transcribe using OpenAI Whisper API
        with open(tmp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        # Clean up temp file
        os.unlink(tmp_path)

        return {"transcription": transcript.text}

    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)# main.py
