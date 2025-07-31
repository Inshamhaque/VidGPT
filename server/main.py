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

transcript = "In the next few videos we'll talk about like packages, static, non static, singleton, class. We'll talk about properties, polymorphism, overriding, constructor and special methods, inheritance types, encapsulation, abstraction, access, control, object class, object cleaning, interface, abstract, abstract, classes, interfaces, generics, exception handling, collection, framework, lambda, expression, enums, file handling, and fast input, output. All of these things will be covered. Nice. Hey everyone, welcome back to another video. And we are starting with a new topic. So we are doing the complete data structure algorithm boot camp. And this is a series that we're starting on now, Object oriented programming. Right, so the next few videos, including this one will be on object oriented programming. If you haven't already subscribed, make sure you do so. And let's get started. So I'm going to teach you literally like everything about it. Okay, so I will teach you like what is objective prompting, what are the various principles specific to Java Stuff as well. And you know, we don't just because content on this you can find in so many books and stuff. So what makes this course special will actually teach you how to think. I will teach you from a beginner's point of view. I will teach you how to master object programming in a simple, easy to understand language. I will teach you what doubts you can have because in object programming people can have so many doubts. So I will tell you, this is the doubt you may be having and we'll then look into it. So it's going to be great. It's going to be a really top notch, amazing playlist. So make sure you like share and subscribe. And let's get started. Okay, so object orient Browning. Okay, before that I want to mention about the playlist itself. So the notes will be provided text notes. So like text files in GitHub repo. Okay, so for notes you'll be having text files in the GitHub repo. And this particular whiteboard stuff, this file is just like rough work. Okay. If there's a concept that requires for me to use the whiteboard, I will use it. So main, main notes can be found over here, like detailed notes. You don't have to make notes yourself. You can use the ones that I provide. Okay, Focus on the lecture itself. Don't make notes, do not make notes because I will be providing those. Okay? So don't. No need to make notes. So this playlist, I'm super excited for this because it's going to be great. Like if you don't know anything about it, even if you know about object, I will literally tell you Everything. And we'll also cover like in detail explanations of why things are working the way they are in Java. Also inheritance and stuff and interfaces and abstract classes and singleton class and generics, also collections, framework, like, and all these other things will be covering. Cool. All right, so let's get started. So over here I'll also teach you about like what are packages and stuff. That will also come. So we are basically going to cover like the introduction about classes and objects, the keywords, like new things like constructors, other keywords like this and final and finalized stuff like, you know, destructors and stuff, wrapper classes, packages, static, non static stuff, singleton, class, the properties of objective programming, like polymorphism overloading, overriding, inheritance, the types of inheritance, encapsulation, abstraction, access control, object class, object cloning, interfaces, abstract classes, generic, exception, handling, collections, framework, lambda, expression, enums. And later on we'll also be doing like file handling, fast input, output and all these other things. This is the entire syllabus for objective Browning. All right, cool. But let's start with like the basic stuff. I'm just going to create another package. I'll tell you what packages are in detail later on. Okay, Introduction. Okay, so what are we doing? Okay, forget about objection bombing. Just forget about it right now. Let me give you a problem statement. You are sitting in a class and someone, let's say, asks you, hey, your teacher asks you, hey, please create a data type. I want to want you to make an application that stores, let's say, five role numbers. You're like, okay, ma', am, no problem. I can just write it in Java. I can say main Java or something. Okay, public, static, void, main. So your teacher asked you, hey, Kunal, please create some sort of a data type or something that stores five role numbers. You're like, okay, no problem. Role numbers are integers. So I can just, you know, I can just create an array, something like this, okay, or you can say new INT5. That's it. No problem. Now your teacher is very impressed. Now your teacher says, hey, can you please create some sort of a data structure or something, guys, that stores five names, right? So you're like, okay, store five names of, let's say your students or something. So you can be like, okay, I want to store five names so I can create a array of an array of string. No problem. String names is equal to new string five. No problem. Now your teacher asks you, hey, Kunal, please create some sort of a data type to store what data of five students. You get confused. You like data of five students? What do you want me to include in that data? Then your teacher says, good question. So your teacher says, include the roll number, marks and the name. Roll number, marks and name. You're like, okay, no, no problem. So you can say something like, okay, if you want the data of students, so you can say data of five students, every single student contains, let's say roll, number, name and marks. Like, okay, how can we do this? So you can be like, okay, let me just create, let's say three arrays. Roll number, right, String. Okay, something like this. And marks can be float or something. Okay, new float. Something like that. Now your teacher is like, you have just created different, different data types for every single property. Okay? So your teacher is like, okay, this is fine. But I want it to be something like this. In just one single line, every single element. For example, in this case, every single element is what? Every single element is a name. Here, every single element is an integer. Similarly, I want some sort of a data structure and in which every single element contains these three things. This is where classes and object oriented programming comes in. Okay, so what is a class? What is a class? Now a class again. Don't write notes. I will be providing notes. A class is a named group of properties and functions. A named group of properties and functions. These three properties that I have right now, roll number, name, and marks. If I want to combine this into a single entity, I can do that via classes, by convention. Class starts with a capital letter, okay? So your teacher will be like, okay, I don't need these three things. I need you to give me something like this, like some other data type called student. Your own data type. You made it. Students is equal to new student. Five students. Every student, like, every single element in this array should contain these three items. So one more simple definition can be. If you want to create your own data type, how can you do it using classes we'll talk more about later. Like when we do encapsulation and stuff, I will tell you in detail what all these things are and stuff. But please focus on basics. Right now, we'll cover everything. Okay? Don't worry. So you're like, okay, I need to put this inside a class. Something like that. Like, okay. And be like, I can basically then create a class. This is how you create a class. This is the syntax for it. Class, class, name. That's it. That is it. As simple as that. Let's say you want to create one student first. So I can say something like student, Kunal is equal to new student. I'll tell you More this new. What about this new thing is and what all these things are? Let's just forget about it right now. Worries student Kunal. Now this, Kunal is going to have this variable. Kunal, this reference variable is of type, student. Our own data type that we created using a class. And it consists of what? It consists of three properties. Don't worry about the static stuff and stuff. I'll cover that in detail. But right now just focus on what we are doing, not how we are doing it. Things we'll cover in detail. Okay, so here it's basically saying that it contains three properties. Role, number, name and marks. So Kunal will have a role, number, name and marks. How to access that, how to modify that and everything will cover later on. Okay, so this is basically what we mean by classes. Now let's talk about this reference variable, object and how this works internally, how to update and stuff and all these other. Okay, so this is basically a template, as you can see. Okay, so a class is like a. It's like a logical construct. For example, this is defining what class is what simply I mentioned to you, it's just a name, group of properties and functions. Your. You can add functions here as well. These are known as methods. We'll talk more about that later. For example, create a method that gives you the GPA of a student. Create a function like that. Okay, so let's look into it a bit more detail. In a bit more detail, a name, group of properties and functions. Okay, what else? What are some real life examples of classes? Car. A car can be a class, okay? And using that class, using that template, so many companies are creating their classes, their cars. There's. It's a BMW, there's Audi, there's Tesla or whatever. Okay, so BMW, let's say Audi and let's say Ferrari. Okay, so these are the three classes. Let me bring up my whiteboard for this. Okay, so I know that class is a name, group of properties and functions. So a car class. If this is a class of a car, okay, a class is just a name, group of properties and functions. Let's say I'm just visualizing it with a box. And what all properties does the car contain? A car contains an engine. A car contains the price and the car contains the number of seats. For example. Okay, now this is a template. This is like the general stuff that every car needs to have. Using this template, many companies have created their cars. Okay, so Maruti created their own car. Okay, Maruti, Ferrari created their own car. Car. Using this template. They're like, hey, this class is a template. Like some properties and functions that every car must have. And please you follow this to make your own car. Okay? Here we have something like Audi. Okay? Very simple. Now what are these boxes that I'm talking about? Maruti 3, Ferrari and Audi, they follow the same procedure. They have these properties. But the value of properties are different. They all have engines. But this engine is, let's say small engine. This one is, let's say big engine. Okay? Or I can say something like, let's say, I can say this is like petrol engine. This is like diesel engine. And this is, let's say electric engine. So all of these three cars have the same property, engine. But the value of that property is different. Okay. Price, for example, 1 lakh 2 crore. 1 crore seats. Let's say over here. 2 seats. 4 seats. 4 seats. Okay. So using this template, they created their classes. Those they created there like cars. Not classes, cars. They created their cars. This is sort of like the template. You can think of it like as a piece of paper someone gave them. Okay? It's like a standard for a car, right? Okay, now please try to think it in the real world sense. Does this thing like really exist physically? Like if you're saying like obviously cars exist, but this template stuff is actually like a sort of like a rule sort of a thing, right? These are the things that exist physically. When you ask, ask someone, please show me an example of a car. Obviously they are going to show you a car, right? They're going to show you something physical. One more class example can be what that we all share in common. Human. Human class. So human class. Humans have some rules. Two arms, like by default, two arms, two legs, two eyes, hair, mouth, nose, ears. By default, these are the rules, right? So some people have like big eyes. Some people have smaller eyes. Some people have a large mouth. Some people have a big mouth. Some people have, you know, ears, different shape. Some people have long hair, some people have short hair. Right? Some people have longer nails, some people have shorter nails. So human class is also like a group of properties and functions. These are all the properties that humans should have. But what actually who are humans? We. We are the ones who are roaming the earth. We have all these properties. But different people may have the different value of these properties. Are you able to understand the difference between the class and what represents the class? These things, these things are known as what? Objects. So a class is what? A class is a template of an object. And what is an object? An object is an instance of a Class. Okay, so as we mentioned about the data type, this can be a separate data type called a class data type. And classes help us in defining this data type. Classes help us in defining this data type. So when you create an object of a class, you're actually creating an instance of that class. What is this instance? Instance means a physical stuff of that class. When babies are born, they are instance of the human class. Human class is a rule. God has provided this as a rule. Like, okay, this is a human class. This is a rule. It does not like. It is not a physical thing. It's a rule. If you want to make a physical thing, you have to have a baby, right? Same thing for cars. This is a rule for cars. Engine price and seat should be there minimum. And in order to make it a reality, the physical stuff that is like follow this instance of the follow this class template and create an instance out of it that is known as an object. So object is what? So class is what? I will explain it in a very nice way. Class is just a logical construct. It does not like exist exist physically. Logical construct and object is what? Physical reality. Reality. This is the thing that is actually occupying space in the memory. Actually is something physically present occupies space in memory, memory, ram, or whatever you want to make. We have been dealing with objects since first lecture. We created objects of what? Arrays, array, list and stuff. All of those are objects only, right? Okay, so class is a template of an object. Object is an instance of a class. Class creates a data type that you can use to create objects. When you declare an object of a class, you're creating an instant of that class. Then a class is just a logical construct and the object is the one that actually has some physical reality that is occupies some space in the memory. Okay, that is the difference between classes and objects. Okay, now objects are categorized by like. It's like they have three essential properties. Okay, State of the object. I am not writing this down because I will be sharing my notes. State of the object, identity of the object, and behavior of the object. So what is the state of the object? State of the object means its value from its data type. That's sort of like the state of it. Identity of the object means whether one object is like, different from other. Okay, it can be useful to think as identity as like where the value is stored in the memory. Right? So for example, you know, you already know this. I don't have to repeat it. Stack, memory, heap, memory, object. 2. Two reference variables pointing to the same object already covered this thing. If you Make a change via this reference variable, original object will be modified. Hence the change will be visible via the second reference variable as well. If they are pointing to the same, please watch the previous videos. Strings, arrays, everything is covered on that. Now I'm telling you why it's happening in previous videos. I told you what is happening. I also told you why it's happening. But now I'm moving into more. More details. Okay. You know, I remember in the previous videos I used to say we will cover an object and bombing. Now it's the time for that. Okay. So this is like the identity and the behavior of the object is the effect of the data type operations. For example, we'll be creating, let's say functions and stuff in your objects as well. So you can say create. You know, every human should have a function called greeting. So every human, when you call the function greeting, they will greet someone. Hi, hello. Something. We'll cover that later. Okay? But this is as simple as it can get. Okay. All right. So why what we are doing? We just covered what are classes, objects and stuff and things like that. Right? Objects are stored in the heap memory and the reference variables are stored in the stack memory. We'll cover more into more details about that. All right. Now let's see how we can make it a reality. So how do we create objects and stuff and assign values, change values. Then we'll add functions into it. Then we'll cover all the other. Let's take our student example. You're like, okay, cool. Kunal, the class is created. A class, student is created. Like this logical construct is being created. Okay, There's a student class and it has some properties like the roll number, right? It has a name and it has marks. So something we have to talk about right now is how do I actually make sure that I create like various. Various objects. Like one student Kunal, another student, Rahul, third student will be like something like Anais for example. Right? So Kunal's role number can be rule number 14. Name will be Kunal and marks will be 87.9% or something like that. Okay. Rahul's roll number can be something like 28. Name will be Rahul. Marks will be something like 92.7. An roll number will be something like name will be an ICE. Roll number will be 33 or something. Random stuff. And marks will be 90, 95.8 or 89 or something. Okay, Random stuff I have just put over there. But the question is how do I make sure that I use this template of the class to create these objects. Okay, before that, I need to actually understand how do I access these things. These class properties are something that everyone will be having. We know that any object of the student class will be having these three properties. Roll number, name and marks. Okay, and we can obviously how do we do this thing, right? How do we create an object? So before that I need to explain to you, like let's say assume. Assume the object is created. Okay? Don't worry about how to create it right now. Assume it's created. Let's say this Object's name is KK. This is name is RR. This name is or not KKRR. I can say this name is student 1. This is a reference variable. Student 2, student 3. Since we have already worked with classes before, this is nothing new. Okay, you have worked with arrays at a list and stuff. So over there, what were you doing? You were saying array, list, list is equal to new arrayless. This new keyword we talked about before, this is used to create objects we'll cover more into like in this video. Now we will cover in detail at the depth of it. Okay, we will cover the depth of it, how it's working internally. But forget about this right now. Don't need to worry about this right now. Right now, just assume these objects are created and these are the reference variables of it. How do we access this thing? Okay, what is student one's role number? What is student two's percentage? What is student three's name? How do we actually access this? We know these students are of what type of classes are nothing but data types. So what are these students type? They're of type student. And student has three properties. Roll number, name and marks. Then how do we access the roll number of student 1, 2, 3, and so on and so forth Using the dot operator. Okay, so the dot operator basically links the reference variable, like the object that. The name that we have over here, Student 1, Student 2, and Student 3, with the name of like the instance variable. Okay, so basically like just the object, the object properties. This one, it's going to link this one, the name of the instance variable, like the name of the instance variable. These ones, these are known as instance variables. What are instance variables? Variables inside the object. They are known as instance variables. Okay, so this dot operator is going to link this with this. So when I do something like dot operator, how do we use this? When I say something like print student one, dot roll number, it's actually linking this object reference variable with the instance variable. So student one's role number is what this actual thing is going to give me. Answer 14. This is known as dot operator. Okay? Commonly we call it a dot operator, but the form in the Java, it is categorized as like dot as a separator. Okay, that's like the formal definition. Okay, but you can call it a dot operator as well. That is it. You can now use this to access any instance variable. So these are known as instance variables. These, these all variables are instance variables. All the variables that are inside your object, they are instance variables. The variable that actually defines an object, something we already covered. Arraylist list is equal to new arraylist. So string name is equal to Kunal. So this list, this name, these are what? Reference variables. We've already covered reference variables before. We've been doing this since, you know, so so long. Right? Reference variables you already know. Okay, so let's talk a little bit more about that, how we can work with this. Okay, so these are instance variables, variables that are declared inside the class, but it should be outside the method and the constructors and stuff that we'll talk about basically just like this. Outside the class. Sorry, outside the method, but inside the class. Okay, so I'll talk more about instance variables in detail after giving the example. Okay, so bear with me for a second. Okay, so here I can say that I have my student class. I've just put it outside the main class or whatever and it has just. I actually did a not so good copy pasting over here. Because it will not be an array, right? It will just be like int, roll number. That's it. String name and float marks. That's it. Okay, because the array that we want to make is of the students collectively. This is a data type for every single student. Okay. For every single student. Like this. Every single student has a role number, name and marks. So if I create a student like this, okay, how do we actually make sure that these things are like modified? So for example, if I try to print it right now, if I say kunal.roll number, try to print that, you can see it's saying, giving me an error variable. Kunal might not have been initialized. Let's talk about this for a second. So in order to create these objects, you have to use the new keyword. Okay, so let's talk about the new keyword a little bit more. Again, it will be again available in the, in the notes. So don't worry about it. Everything will be, you know, given in the notes, so you don't have to like really make notes or anything else. Okay, let's Talk about the new operator. What does the new operator do? So if you have a student class like this, if you have a class student, it has three properties. Int, roll number, string name. Now we're talking about how to actually create objects and float marks. Okay, this is what you have. This is a class. How do I create objects from this? You know, this is a data type student. So it's going to be obviously student over here. Like we do arraylist into arraylist list. You can give a name to the student. I can say student number one. This is my reference variable. Okay, now in order to create an object. Now what is this thing? This is actually known as like you are declaring the reference to the object. Okay, so you are here declaring it. Declare. Declare this reference variable to this object to do to an object of this type. This is actually not creating an object. Okay, the first, this line that we have over here, this one, this is basically just a reference to an object of type student. At this point, the student number one does not yet like refer to an actual object. Okay? It is just in the stack memory, like this stack memory, we have the stack memory, we have the student one. That's it. Some people may ask Kunal, okay, this reference variable is in the stack memory. You have not created an object. Whatever you have just declared it right now. Then what is it pointing to? What is it pointing to? Before asking such questions, try to print it. You will find the answer yourself. Let's see, let's try to print it. Okay, so let's try to see if we try to print this, what will happen Right now by default it's not going to. It's just going to say that it is not initialized. For example, right now. Okay, that is what it's going to say. So when the, when, when it is like not initialized, it will ask me to hey, initialize it. And by default it will do it as null, for example. Okay, but let me do something else. Let me say that okay, this is a, this is a. This is an area of students, right? This is an area of students. So let's try to do one thing. Let's try to see what is inside this area. If every element in this array is a student class type and what is the type of that thing? Print arrays dot to string students. Try to print and see what we get. Null. So when this is not initialized, the student. When it is not initialized, by default the value is what? Here in Java, null for primitives, it may be different. We'll talk more about that later. But for objects it's null as you can see from this example. Okay, all this, all the students here are null. Now let's talk a little bit more about that. Like how we can actually create this object and stuff. And by default if it's null. So here you can see that this is declaring. So this is actually just, you know, we are just creating a reference of type student. But if we want to actually create an object of this, we do it like this. We have done this before as well. Student one is equal to new type student like this. So this new operator, it. It actually dynamically allots. What do we mean by dynamically? A lot. Very important point. Allocates the memory at runtime and returns a reference to it. Okay, so let me write this down. I'm telling you everything. How things work dynamically. You are responsible for working with the notes and everything. Okay, I am putting out knowledge out there. Dynamically allocates. But I'll mention this on my notes as well. Dynamically allocates memory and returns a reference to it reference variable to it. Okay, which is going to be stored here in student one. Ok? A reference to it. Okay, that is it. And this reference is stored in this student one variable. That is how it works in Java. The student, the objects are stored in the memory heap memory. So this is the heap memory. An object of this student type will be created like this. Okay. And it will be pointing to this type. And this is going to have role number, name and marks. Something like this. If you create another student student number two it will have another object student two like this. This is how it works internally in Java. Very important point. Please note in Java this is how this thing works. Hence. Hence all class objects in Java must be allocated dynamically. What do we mean by dynamic dynamic dynamic memory allocation? Some people may not know that if you are not aware of dynamic memory allocation, it means you have not watched the first video of the DSA boot camp Introduction to programming over there. We covered it static and dynamic languages. I mentioned five, five points. I mentioned stack and heap memory over there as well. Do you want me to repeat it? I will repeat it, otherwise you will say Kunal, you don't teach. I have taught it in detail in the Introduction to programming lecture. Dynamic memory allocation means that first you know why am I telling this? Do you want me to explain to you how at the entire Java code compiles? No. Watch the Introduction to programming Introduction to Java video over there. You will understand what is dynamic memory allocation. But still I'll give a brief about it. So when you write something like this, student, student, one is equal to new student. I'll talk more about the new keyword and what this, this thing is. Every single character that I'm writing, right? I will tell you about that. Every single character. I will not leave anything. That is why this is going to be a long playlist. So here, here it's saying that you know all the things that are on the left hand side of your equal to. This happens at compile time, this happens at runtime. What is compile time? Compile. You all know what compilation is because you have been following the course. I've covered it in detail. So compiling means that the program is checking whether there are errors or whatever like that In Java, you know, it gets converted to bytecode, then from bytecode it goes into machine code and stuff using JVM and stuff and all these other things. And runtime actually means when the everything is finished having happening, all the checks are happened, all the, you know, code has been converted. Runtime as the name suggests, means the program is now running, your application will be running. That is when the memory allocated. Memory will be allocated. That is known as dynamic memory allocation. Like memory in the RAM or whatever will be allocated when your program is running. That is what happens and that is how new works. Okay, enough details about it, please make sure you check out the previous videos. You will understand even more. But I think I already covered it in detail. Okay, and stack, memory, heap, memory variables are reference variables stored in the stack. They point to the heap, memory, heap memory also in detail. I have covered this like all these things before, but I don't think anything is left out. I literally revised the concept from scratch. Okay, so that's it. Yeah, you can use now the student one mimicking it like as if was the object. But it's not the object, it's a reference variable pointing to that object. In reality it is holding like internally. It may be holding like some memory address or whatever of the original object that is in the heap. But we can't get that memory object. We can't access the memory address because in Java it's not allowed. Okay, so this is the key for Java safety that you cannot manipulate references as you can with like pointers in C and stuff. Here you can't do it. You can't access the address, it's not allowed. Okay, so you cannot cause an object reference to point to an arbitrary memory location or manipulate it. You know, like for example an integer or whatever. We have already you know how integers work, right? And we'll cover that in detail as well. Okay, so if I'm saying we cover that in detail, let's do it right now. So let's look at this thing, what it. What it basically means. So here if I'm saying that the student one is equal to new student. So this is actually a variable that is of type student of this class being created. And the class name is the name of this class that is being instantiated. Instantiated. So when you create an object of a class, it means that it is being instantiated. Okay, so one more thing remaining now is I've detailed in detail covered news or this new. If someone asks you what does new does dynamic memory allocation, that is what new does. Okay, Only thing that I haven't explained right now is what is this thing? This is something I have not explained right now. And I am not going to leave out any detail. I will not let you have any doubts. I will cover every doubts in the videos itself. Still, if you have any doubts, I'll actually teach you how to study object as well. Just like I taught you in recursion. I will teach you how to study object and programming as well. Okay, so what is this? This thing. But right now you know how to create new objects. You know that. How to access the variables of that object. You know that. Let's see how we can now manipulate it. Okay? After that I'll teach you what is this bracket thing. Okay, no problem. So now you are saying something like if I'm declaring it. Okay, so now I have declared it and now I'll initialize it. Kunal is equal to new of type student. Forget about what this thing is. I will tell you right now. Okay? Right now I will tell you in detail. But for now don't worry about it because it's coming in one line. Also you can do in one line. You can do it like this. In one line you can say student Kunal is equal to new student. Something like this. Okay, let's try to print the by default value of Kunal. What it is. It's giving me some random value. Okay, this random value, I'll. I'll cover that in detail. Don't worry what this value is and everything. So. And actually how to print some pretty values. I'll cover that in detail. When we do object, like when we do polymorphism. Right there, I will cover it. Okay, so right now let's say it's just giving some definition of some object value or something. It's like some Some value it's giving random value. Okay, for now. What if I try to print kunal.roll number. I will cover this thing as well. System out, print LN and stuff. By default it's giving me zero. By default it's giving me 0. Let's say if I try to print Kunal name, what will. What will it give by default? What is the value by default of string type when it's not initialized? We just did it. What is the by default value of objects when they're not initialized like this? Null, right? By default will be null. These are primitives. So these will have by default values of an integer is 0 of Boolean is what I think. False of float is what. Let's see. I think it's 0.0 or something. Kunal marks by default value of float is what I think since it's primitive 0.0. Yeah, that is correct. Very simple stuff. For all the other primitives you can try it out. What are the default values? So in short, already covered it previously. Integers have some default values. But the objects on the other hand of more complex data types. Since this is a class because it starts from capital letter, you can also control click and see on it. This is a class, right? This is of type like string. So obviously it's not initialized. Right now it will be of type what? Null. Okay, so I can modify it according my needs as well. So I am accessing this reference variable via RNO so I can say Kunal's role number. I am accessing it right now. Fix it to 13kunal name. Fix it to Kunal Kushwaha Kunal Marks. Fix it to 88.5%. Okay. Float run this make sense. Very easy stuff. Let's say you don't have it like this and you have some default values available over here. Let's say some default value like 90%. In that case, let's see what will happen. 90%. I will teach you why this is happening. What is happening with a diagram. You will never ever get confused or forget it into in your entire life. My guarantee. Let's bring up the whiteboard of how this thing is working. Okay, so student created like this. Okay. Student class. Student class. It has a roll number, it has a name and it has a marks. By default this is equal to what internal Java implementation. It says primitives like integers are zero by default. This is a string name so it will be by default null. Okay, I create an object. I say using new I create an object like this. I'll do it like this. Okay, for example, right now, here, now it's going to say let's say this object name is kunal the reference variable name. Okay, so when I do something like this, when I say kunal.rno role number first it's going to check in the object. Does roll number exist over here? Please understand, I'm visualizing it for you. It's going to be like hey does roll number exist over here? It's going to be like no it does not. Okay, look at what the default value is. Then it's going to be like default value is zero. It's going to print zero. Then when I say something like kunal.name is going to hey, does name exist over here? It's going to. No it's not. Okay, print the default value null. Similarly, print the default value for marks. But when I do something like this, when I say kunal do roll number is equal to 13 Kunal do roll number, this kunal dot means in simple terms it is not. It may not. Yeah it is. In simple terms it means the Kunal object.roll number means the roll number inside the kunal object. So it's going to be like I'll make a roll number over here and the value of this is going to be =to13kunal. Name is equal to kunal kushwaha. Okay, name hover here is going to be kunal.name is going to be equal to kunal kushwa. Let's say this is all I do now. When I print kunal.rno it will be like hey does RNO exist in the object is going to be like yes it does. Okay, print13kunal.name does name exist in the object? Yes it does. Kunal Kushwa. Print kunal.marks do marks exist in the object. It's going to be like no it doesn't. It does not. That's like okay, print the default one. Similarly, when you create a new object like this, you create Rahul, you don't initialize, you just initialize it. But you don't actually change the values of the reference variable. The these variables and stuff that we have over here. So in that case what is it going to do? It's going to be like hey give me Rahul roll number. It's going to be like does Rahul roll number exist over here? It's going to be like no it doesn't print zero. So.rollnumber. this thing whenever you do from a reference variables for non static type, I'LL cover what is static and non static later on. Forget about static and stuff. Right now. Right now, just imagine that every object had its properties that are like common to this thing. Okay? But here you cannot add like some random property. You can only do Kunal rno, Kunal name, Kunal Marks for only those variables that are already present in the template of the class. You can't do something like kunal.salary. why? Because salary is not available over here. Okay, let's try it out. You can do it in Python. Okay, you can't do it over here. This is a static type language, Java. So if you try to do something like Kunal Salary, it's going to give an error. It's like, hey, salary does not exist in the type of Kunal. What is the type of Kunal Student does not exist over here. You can do it in Python though. Okay. Which will cover in the Python boot camp. So that's it? Yeah, that is it. As simple as that. But this is a little bit cumbersome. This is a little bit cumbersome question. Doubt you may have Kunal. Can't you just allocate these values while you are creating an object? Like when you are creating an object like this, can you not just like allocate the values during that time? Because it's getting very repetitive. Kunal.roll number, Kunal.name. let's say Kunal had. Let's say student had 100 properties. Would you write this hundred times? Isn't there a better way? Yes, that's where this thing comes in. What is this constructor? This is known as a constructor. It's actually a function type thing because we are calling it. So it's a special function. Let's see what a constructor is and how it works. Okay, so constructor, basically, I'm not going to write this in this whiteboard because it's very simple statement. A constructor basically defines what happens when your object will be created. Okay? Hence this is what we call by this is what is simply what. And what do we need to do when the object is being created? We actually need to allocate these things. We need to allocate these things. All right, so let's see how we can do that. So let's look into it. Let's like how are we going to use the constructor to basically instant, like instant, instantize, instantiate. Okay, instantiate. Okay, initialize or something. Okay, not initialize, but actually create an ob. Create an object out of it. You know what I mean? Okay, so Basically what we want to do is put some values over here. I think that makes more sense. Easy to understand. Okay, so instead of having these default values, let's see how we can use a constructor. We already know how to do it. We are doing kunal.roll number set something. Kunal name set something. But let's say we might want to do something like this. Do it once at once. Make it mandatory to do. Okay, it's not like you are only setting roll number and name and forgetting the marks whenever you're creating a new object. No, put all the three items. Let's see how we can do that. Okay, so constructor what it, what it is and how we are doing it. So basically whenever you. This is the. This is sort of like the. No, this is sort of like the. Our constructor. Basically like looks like. Okay, for example, so if you're saying something like a constructor, so you're saying something like let's say student. So what we want, let's say we consider the what first and then the why later on. So don't worry about how it's happening. Let's first see what do we want. We want to do something like this. When you are creating an object during that time, initialize all these like are the values of all these variables that the class has. Okay, no problem. This is what we want. We want to do something like this. Student Kunal is equal to new student and we pass the value inside it. I can just write it in new line is equal to new student. Pass it in a new line. Here I can give a role number, name and marks. Now when I do kunal.name it will give me 13. Sorry Kunal roll number. Give me 13. Kunal name, Kushwaha and Kunal marks 84.3 or whatever. So this is what we want to do. Now how this thing works internally. This you can see you're calling it and it starts with the class name. This is actually a special type of function. Special type of function in class, in the class. Then you're going to be like hey Kunal, when you were doing it previously like this, when you were doing it over here, we did not create any function in the class. We only had three things. Roll number, name and marks. Then what is this function being called? You are saying this constructor is a special type of function that is inside a class. But I don't see you creating any function inside the class in the previous example. Then what is this function being called? This is known as the by default constructor okay. When you don't have any constructor inside the class, constructor was a function. Okay. When you don't, constructor is a function that is run. Let me just write it down. Constructor is special function. So no confusion over here. It's a function. What? What else do you want me to say? It's a function, special function that runs when you create an object. When you create an object and it allocates some variables. Okay, as you like it. Okay, so if it's a function, it will be having some arguments right here. By default it has no arguments. This is known as the by default constructor. You don't have to make it or anything. By default, if you don't have a constructor, Java will call its own constructor that is just go going to assign everything empty. Just like we did right now. Everything was empty. You can see it was just printing 0, null and 0.0. That is a by default constructor. No need to make it. What if we want to make it on our own? Let's say during calling the function we want to give it three arguments. Roll number name and what? Roll number name and marks. And we want it to bind these. Bind these arguments with the object. Bind these arguments with the object. So what do we want to do? This is a very important point. Otherwise you will get confused. Nothing new. We were doing Kunal name is equal to Kunal Kushwah outside everything. Okay, let me show you. Let me show you what we want to do. So we are doing this thing. Kunal.name is equal to Kunal Kushwaha or something. Okay, Kunal.name is equal to Kunal Kushwah or something. And let's say I do Kunal dot marks is equal to 84.5 float or something like that. Okay, this is what we were doing. Or I can just copy uncomment this out. Okay, now we want to do it inside a constructor. How do we do it? Let's say we do something like this. We say instead of doing it here, I will do it inside the constructor. The constructor is somewhat like this. It's a function. So I would have to create a function. I will just say student. That's it. This is a constructor. You don't have to add a return type or what? You don't have to add a name or anything like that. The return type is the class itself obviously. Because it's going to return the create a type of the object of this class. Hence this is like the return type. But Name is not required. That is it. Okay. What it is public or private or whatever. That is something we'll cover when we do. Access modifiers. Forget about that right now because there's a separate long hours and hours long video on that access modifier will be covered in detail. Right now just focus on this simple stuff. Return type is obviously this because we are creating an object of type this and constructor is a special function that creates an object. No problem. So this will actually run when you create a new object. This is what we want to do. Giving me some errors. It's like okay, you want to set the Kunal's role number equal to 13 or whatever. That is what you want to do. But what is Kunal? I don't know. This is a template. In the template in the rule section, how can you mention just Kunal? You're like, hey, I'm just referencing the reference variable of the object. This is the reference variable that I've created. I want you to add this reference variables roll number should be equal to 13. So like. But why are you adding Kunal over here? What if you want to make another one called Rahul, Would you make another constructor called Rahul like this or Rahul another constructor or something like this? Then another problem will be how would it differentiate which one to call. So what we need to do is we need some sort of a way to initialize these. I'll just write it over here. We need a way to initially like to add the values of the above properties object by object. Like for every object it will be different object by object for that. So we need to access. We need, we need one word to access every object. What do we mean by that? So when you're saying student Kunal is equal to new student, we need it to automatically replace this particular space with kunal. When you're saying, this is a very important point. When you're saying student Rahul is equal to new student, I automatically wanted to put Rahul over here. When you're saying student and ice is equal to new student, I automatically want you to put an eyes over here. Then it will automatically if you put an eyes over here. What we are doing, don't worry about how we are doing it. What we are doing. So when you automatically put an eyes over here, it will automatically set an eyes dot roll number is equal to something. And I thought name is equal to something. When you automatically put Kunal over here, it will set Kunal rain roll number something kunal.name something. What is the Keyword for that to access the variable like that. The reference variable, the object. It's known as the this keyword. That. This keyword. Okay. You do not have to confuse this keyword with anything. It's very simple. What is this keyword? This keyword basically means when you are calling this thing. Let's try to print what will happen. Let's see. Let's see what will happen. 13 Kunal Kushwaha. 88.5. What happened? Debug. Let's try to debug. Let's see what happened. So you are over here. It will go inside. Nope. We have to do it again. Want to go inside? It will go inside. Where Constructor. Whenever new object is being created, it goes inside the constructor. New object being created. Okay, go inside the constructor. Now this is not the default constructor. Because the number of parameters here are none. And this constructor also has a number of parameters none. Hence it's going to now call this one. It's go inside this. It's going to say over here. This dot you know roll number is equal to 13. This.name is equal to Kunal Kushwah. This dot marks is equal to 88.5. What is this over here? This. This. This is going to be replaced with this thing. Kunal. Okay. This thing, this orange color. This will be replaced with Kunal. Internally this thing will take place. Kunal rule number 13. Kunal.name is equal to Kunal Kushwaha. Kunal.marks is equal to. This comes out of it. Now it's just printed Kunal dot name kunal.roll number kunal. So please don't get confused with this. You. If you are thinking that it's confusing then you will get confused. It is not confusing at all. This basically is just doing this same thing. This thing. That is what it's doing. But it's much more general. When you do something like this student Rahul is equal to new student. In this case it will say Rahul dot name is also equal to Rahul dot roll number is 13. Rahul name is equal to Kunal. So class is what? Class is just a template. So in order for you to access the object inside that template like okay, you are doing something. You can also let's say add functions inside the class. Okay. Okay. Let's try to add some functions inside the class. I create a function greeting. Hello Name. It just needs to say hello, my name is name. So I will just say okay, no problem. Let's. Let's do that. Function Function we know we'll just say void, return, type nothing greeting and it's going to take nothing. It's just going to say print. Hello, my name is. My name is name. Okay, something like that. No problem. No problem. Now you will understand the meaning of this. Now you will understand the meaning of this. So now if I try to print something like this, I say Kunal dot. What? Kunal has a greeting method. Yes, it does. Kunal greeting. Pause this video. Think about what will print. Think about it. Is it going to print hello, my name is Kunal or hello, my name is something else. Let's try to see. Because you have already set Kunal's name is equal to Kunal Kushwa. Over here you have set it. It should print Kunal Kushwa, right? Let's see what it will print. Hello, my name is Kunal Kushwa. Okay, but one more thing I want to mention over you is that what if you try to write it something like this, this dot name. Because we know that. Okay, I will tell you one single word on this. The output will also be same. But I will tell you one single line that will clear the meaning of this for you. Okay, My name is Kunal Kushwa. One more doubt you may be having is Kunal. It's giving the same answer with or without it. Then why? Why are you doing it? So let me explain. Internal implementation is. By the way, this basically means that. Okay, whenever you try to access any particular item of the class via its object, obviously every single item will be specific to that object. Greeting for everyone will be different. Name and roll number and marks for everyone will be different. In order to do that, we use the dis variable. So when you do something like this kunal.greeting internally it's saying that okay, hello, my name is. Whenever you say reference variable name. Whatever is inside the class, this is replaced with the name of the reference variable. So when you do kunal dot greeting it will be like hello, my name Iskunal.name. that is what it's doing internally. When you call the constructor internally, it will be like okay, constructor is being called. Kunal is like Kunal is something we are creating. Kunal is equal to new constructor. This will be Replaced with Kunal Kunal.roll number is this Kunal. Name is this Kunal dot marks is equal to this. Okay, that is how it works internally. The doubt you may be having is why is working this or with or without it? I will clear that right now. Okay, but internal implementation Is imagining that this dot is already present over here. Okay, if you have done Python, it's like similar to self in Python. The self keyword, that is basically what this is. It's just replacing it with the name of the reference variable which is Kunal. So as you can see, you can add functions over here as well. Let me say one more thing. Let's say I create a function void, change name and I pass a new name over here. New name. Okay, I'm saying that name is equal to new name. Something like that. Okay, I say Kunal do change name now. Okay, I will say Kunal change name. Kunal change name to something like what Shoe lover or something like that. Now when I do greeting, let's see what will happen. My name is shoe lover. Okay, so you're just changing the name. But the important thing over here is that you have to write like this over. So for example, you do something like this. Let me show you one more thing. Let me show you something like a constructor. One more type of constructor. Then it will make things much more clear. And when things start to break. Okay, so let's say you don't want to. Because if I'm creating let's say for Rahul right now Rahul also have the same properties. Because every constructor is providing the same property. I don't want this. I want to add some properties inside. I want to say okay, pass a roll number, pass a name and pass a marks or something like that. Then I can say this roll number is equal to the one that you're passing over here. Roll number. This name is equal to the one that you are passing like this. This float is equal to the one that you're passing like this. Imagine I don't put marks over here. Okay, imagine I don't put marks marks over here. Then let's see what will happen. Now if I say something like Kunal is equal to new Kunal, I will add roll number 13 Kunal Kushwaha as a name. Oops. Kunal Kushwaha as a name and 85.4 float as marks. Now if I try to print it, let's see what will happen. Let me change the rule number to something else. Run. Actually did not print it. Let me just hide this thing. Let me print it like this. All the three things, let's see what we get. It did not get modified. Did not get modified. Okay. And by the way, this name and these, this name, this variable name and the one that you have inside your class. These do not have to be same. Okay. That is why we are putting this over here. So this can even be role. Now. Now it's working. Okay. Name or NAAM or something like that. Now we have something like naom. So Java is smart. Java is like if you're taking N over here, this NAAM is naam. But if you had the same variable name. Okay, if you have the same variable name, in that case you have to do something like this. This dot roll number, this dot name, this dot marks. Now internally it will be something like kunal dot roll number is equal to rno. This RNO does not have a dis associated to it. So this will not be kunal dot rno. This will be the RNO that is passed over here. Now if I print it, see what will happen works. And this does not even have to be like this. You can say roll number. Like roll number. This will also work. Why? Because reference variables. Right? This is also working. This is going to be having some internally if I show you, it's just something like this. It's very simple. So internally if I show you something like this. So it's like in the constructor when you pass something like role. So this role is going to have a value of 15 and kunal.rno is going to be equal to role. So it's just going to say kunal.rno is equal to 15. That's it. That's how it works internally. Okay, let me tell you a bit more details Like I still remember like what you're saying. You might be having more confusions about why this, this is important when it's working with and without it. Okay, let's see how. Basically it's a good convention to put this over here. Internally Java is smart. It's assuming that. Okay, you are saying this only. But imagine the name, the new name variable in the local variable name was also named. This will give you an error. Now it does not know which name you are talking about. Are you talking about this name or are you talking about this name? That's why you have to specify. So this, this is basically what this. This basically means Kunal. If you do Rahul. So with Rahul you can say roll number is 18 name is Rahul Rana and 90.3, something like that. Now when I print Rahul something it will print the Rahul's names and stuff like that. So what is this? This is just replacing the these keywords. Okay, cool. Let me show you one more example. Okay, this is what I was talking about. I used the Name example. So I realized my whiteboard was on. So if you don't have this keyword over here, then it will be like are you talking about this name or are you talking about this name? Well, I know kunal.name. set it to whatever name you have passed. This can be anything. Nam Naam. No problem. But by convention it's easier to put it as same. Okay, so a hint of like some object principles. When you call a constructor with three values, it will call this thing. When you call it with zero values, it will call this thing. This is known as function overloading. Constructor overloading. So if I create a random person, if I create a random person, student, random is equal to new student. It's going to be like, hey, it's calling like an empty one. So do we have an empty one? It's going to be like, yeah, we do. We have an empty one. Let's say you remove it. In that case you will get an error. It's going to be like an empty one does not exist. The three argument one exists. Okay? Similarly you cannot just add two arguments or whatever. Now it makes it mandatory for us to add arguments over here or okay, I'll cover it in detail how the memory management for this works. Polymorphism. This is known as polymorphism. I'll cover that in the polymorphism video. But let me say something like this. Let's say that I want to create. Let's say I want to create another object, another constructor that actually takes value from another object. Okay? So student, it takes values from another type of student. Okay, Other student. And I want my current name to be equal to the previous name, other student's name. So here you can say this dot name is equal to other dot name. You can say this.roll number RNO is equal to other.roll number. And this dot marks is equal to other dot marks. So here you can say something like this. Here you can see that if I pass Kunal over here. So this other will be replaced with Kunal and that this will be replaced with random. Here you will have Kunal. It will literally replace Kunal over here. Here it will replace random. Random.name will be equal to kunal.name random.rno will be other.rno. okay, now if I print random.name or random.name. if I print it random.name. let's print it. Kunal. It will give me Kunal Kushwa. Okay, you can remove it like this as well. But then good convention is to follow the this naming convention that this dot. Okay, because we see it was giving an error over here when we were doing like the recursion. Sorry, not the recursion. This thing, it is giving error over here. Like right now it does not know if this was also rno. This was also rno, it would not know which one to put. Hence it will just lose the default values of this. That's why this keyword over here is important. I hope you are able to understand what this is. Right. This basically means what object you are referring to in that particular thing. It will make much more sense also when we do static. Because. Static. Forget about it right now. Ignore. Ignore. I said static. Okay, we'll cover that later on. Okay, that is basically what this keyword is. And that is basically how you make multiple constructors. Now you can use this constructor, pass the values and create as many people as possible. When you say something like student, ARPIT is equal to new student, something like this 17arpit 89.7 here this will be replaced with arpit. Hence internally it will be something like this. Arpit roll number is equal to the roll number you have passed. Arpit.name is equal to the name you have passed. Arpit.marks is equal to the marks you have passed. Same thing like we were doing here. This thing now it's just happening internally in a much more streamlined way. This is a correct way of initializing objects. Okay, that is basically it. We covered about this as well and. Yeah, that's it. Let's look into a few more examples. Okay, that was about like constructors and stuff. They don't like really have a return type, because implicitly like the type of the class itself is the return type. So no, no, no void or anything you have to write. And the disk keyword also, you know, sometimes a method in the class, you know, it would have to refer to the object that invoked it. Right? So that is why we had the. In the method. Also I showed you the change name one, the greeting one. Okay? Sometimes like if you put the same value I was. I also showed you that it was giving not the correct answer. So this keyword is nothing. It's just replacing the this with the name of the reference variable itself. So that way it can say kunal.name, kunal.something. okay? That is why a general way for putting the name of the object is this because I can't put Kunal in the template of the class, right? Class is a template, right? So for that, Java itself defines this, this keyword. Okay? So for it can be Used like inside our method or anything to refer to the current object. Okay. It will be replaced with the current object when you call it. And yeah, that is always like, you know, this is always a reference to the object on which the method what was invoked. Like we did right now. Okay. And constructors will. Constructors are not done right now because we have to learn about constructors in Inheritance as well in like classes and derived classes and stuff. But constructor, you know, in short, like we already covered it once, you know, we defined a constructor is automatically called when the object is being created and obviously before the new operator that we have completes. Okay. And constructors, we talked about how they work and everything else as well. And yeah, these are the ones that are actually, with the new keyword, they're actually creating an object in the memory. Okay, so let's look into a few more things and then we'll wrap. Okay. One more thing that this variable can help us in doing is you can call a constructor from another constructor as well. Okay. So you can do that as well. Let's say this is an empty constructor. It does not have anything. Let's say I'm just going to remove everything from it, right? So let's say this does not have anything. And whenever you, let's say call an empty constructor like this, let's say whenever you call an empty constructor like this, let's say student, random 2 is equal to new student. Let's say whenever you call a random one like this, it should automatically call. It should automatically call this default one. Okay, Sometimes that might be possible. So you can say call another constructor. This is how you call a constructor from another constructor using the this keyword. What is this? This is a type that type of that class itself like this. So it will be calling what roll number I can add by default. Let's say 13 string name can be default person and marks can be default, let's say 100% or something like that. So this, this basically refers to this constructor. So when you call this, it's actually going to say. Well, you can't really call the, you know, can't really call Kunal like that, right? Or whenever you. Basically it's not, it's just creating a new object right here it say creating a new object. So inside this, this is going to be replaced with student, another constructor. You can't really say random to call. Why? How can you call that? Random 2 is a reference variable. So in this case, this is actually replaced with the name of the class. So internally it's something like this. Okay, internally it's something like. Internally it's something like new student this thing. Internally it's something like this. Sometimes this is what you want to do. Like some people, they pass nothing, but still you don't want to do it. You want to pass some default items. Now if I print out like random2.name, it's going to say print random2.name. Okay, so if you print random2 name, it's just going to print this default person name. Okay, default person name. So that's one more use case of this. You can use it to call constructor from another constructor. As simple as that. Because you tell me, if I tell you that this actually reference to the reference variable. It does in all the other ways when it reference to the reference variable when you're calling it via a reference variable like kunal name. Kunal greeting Kunal, change name. But while you are creating an object and then use this, then it replaces it with this thing. As simple as that. Okay, that's it. That's all you need to know about this. And we'll be covering it later on as well. But I believe this is clear. Right, let's move forward. Okay, let's talk a little bit more about some other things that you may be having a doubt for. So you may be having a doubt. Like you might wonder like why we don't use the new keyword for creating. Like when we do, let's say primitive data types like integers or characters or booleans or floats or something like that. The answer is that in Java the primitive data types are not implemented as objects. I know that objects are stored in the heap memory. I told you this previously as well. Okay, but in Java, primitives are not objects, hence they are stored in the stack memory only. Now you will say, hey, Kunal, in the introduction to programming course, you gave an example of A is equal to 10 and a 10 is pointing in the heap memory and A is pointing in the stack memory. You may say that, and I stand by my statement that that was correct. Why? Because in that I specifically mentioned 10 to be an object. That intro to programming video was not specific to Java. For example, in Python, even primitives, there are no primitives in Python. If A is equal to 10 is something you're writing in Python, 10 is an object in Python. Okay, so in that video I specifically mentioned that 10 is an object in that case and that objects are stored in heaps. And in the later video, when we did data types in Java, primitive data types in Java there. I also mentioned that primitives are stored in the stack memory. Okay, so they are implemented like primitives. To answer your question, why we don't use new the why we don't use new keywords. So because they are implemented like as normal variables, we don't implement this, those as like objects, they're just like primitive data types. Why do we do it? Why does Java do it? To increase efficiency, to make it more fast. Because putting into the heap memory have big, big objects and stuff, it will be much more slower. That's why. Python is also slow language by the way. Python is very slow, which we'll talk more about in the Python boot camp. Okay, so we talked about it, but we also talked about like, you know, how it allocates memory like during runtime. So yeah, I mean, let's talk a little bit more about that. Just one more thing, one last thing I want to talk about is how it allocates like memory at like the new keyword when we're talking about at runtime. Stuff. So when I write something like this. Okay, okay, imagine when I write something like this. When I write something like student one is equal to what New student. Okay, no problem, no problem. Then I write something like this, I can say student 2 is equal to what? 1. Now obviously both 1 and 2 are referring to the same object. Why? Because you created student number one. You created its object in the heap memory and now it's pointing to the heap memory. Then you created student number two. It's pointing to the object that two is pointing to, hence the same object. We have covered this in detail. Like when I showed you how two reference variables are pointing to the array list in the recursion also, right? So many reference variables pointing to the same object of the array list and it's changing the entire stuff. That's exactly what is happening over here. So this assignment of one and two, this assignment that we are doing, it did not allocate any memory or like this assignment of 2 to the variable 1. It did not like allocate a memory or like copy any part of like the original object or anything. It is just simply saying point it to the same object as the pointer number one is doing like the reference variable number one is doing. Okay, so obviously as previous example showed, any Change made via 1 will also lead to changes via 2. Okay, so when you assign one object reference variable to another object reference variable, you are not creating a copy or anything, you're just making a Copy of that reference variable only. Let's see it in example. Okay, so an example for this can be something like this. Like if you make a change via one reference variable to another one, the normal volume will be changed. So for example, if I make, let's say random or I can make one or two only. Okay, so, so student one is equal to new student. Okay. And then I can say student two is equal to one. Okay, let's make a change in this particular one. Okay, I can say one dot name is equal to something something. Now when I print two dot name, it will also be changed to something something. Let's see what we get. Let's see something something. So I changed the name of one, but the name of two was also changed. Why? Same thing. Like as I mentioned, over here you make, you make a change via one, you can say one dot name. One dot name over here is something something. So then when I access two name, it will also be something something. But we have covered this before in like the arrays lecture. Let's move forward. Okay, so we talked about like the primitive data types and stuff there. Also there's also a way to create it using the new keyword, okay, that is known as wrapper classes. Let me show you how we can do that. So here you can see if I create another class called let's say wrapper examples. Okay, Something like that. So if I say public static void main you can say int a is equal to 10, but you can also do something like this. Integer num is equal to new integer a constructor you can pass 45. Okay, it's actually not required. You can directly do it like this as well. That will also create like an object. But now the difference is what is the only difference. Now it's being created like as an object. Okay, so it will be like something like. Like if you do a dot, you will not get many stuff with it. But if you do num. Now this num is actually an object. It's not like primitive stuff. It's actually an object of type integer. This is known as a wrapper class. So obviously it comes with its own prop. What is a class? Properties and functions. So it has so many properties. Byte value compare to double equals, float value, hashcode, tostring and all these other things you can use. Okay, so now you can see you got so many functions okay, for it. That's basically what like the wrapper classes are about. So basically converting the character into a what, like an object, like a primitive into like using it as an Object. But one more thing you will say is that Kunal, you made a. You made a program previously which was known as swap. So you had something like int A and int B. And when you swap it, A, you can say B is equal to in temp is equal to A, A is equal to B and B is equal to temp. Something like that. You are saying that Kunal, when you call this swap function, if you say int A is equal to 30 and if A is equal to 10 and I can say int B is equal to 20, int B is equal to 20. Like this, okay? And when you say like I'm going to, I'm going to swap A and B, it will not swap. We have already covered this before. Okay? This has to be static. What is static? We'll cover later on. So if you swap it, if you just swap a comma B, then you print A plus B. It will not swap it. We already know it. Okay? We know that it will not swap. Why? Because these are primitives and in Java there is no such thing as a pass by reference. Everything is passed by value. So in primitives pass by value is just a value. So A, the, the pointer to this, the reference variable will not be passed, just 10 will be passed. So this A is not actually equal to this one. This is an A that is in the scope of this function only. So it's swapping it inside this function only. It's not actually swapping the original one. I already covered this before because Java is passed by value. But when you pass objects, the reference value is passed. References value is passed. This is also something we already covered, no problem. So if you convert it to something like this, integer A and integer B something like this, and you convert it to something like this. Let's say I just convert it to something like this. Integer A is equal to 10, integer B is equal to 20. Now this is object. These are reference variables. Okay? Now so obviously this will not be like passed by reference. Obviously this will be passed by the reference value. Should it swap now or not? Let's see. It won't. It's still not swapping like Kunal. Now what is happening? You said this is like passing the reference variable to it. That is true. I can also create a new. Let's say this one. This one can also be of integer type. No problem. Not swapping. Why? Let's see. If we take an example. If we take a look into what the integer class is, you can see it's a final class. Okay. This is the reason. Let's look into final. Let's see what final is and we'll cover more about classes that are final later on. But this is basically the reason for it. So basically when you type like integer a something like this, so here it's just saying, okay, I create a temp. Temp is going to store the value of A. A is going to point to B, B is going to point to temp or whatever. Here also it's not actually changing the original values because it's like final and stuff. Okay, let's see what final is. That will make much more sense. Okay, so what is final when we talk about final? So final is basically like we talked about the keyword. This keyword, Final is a keyword and using that keyword you can basically prevent for your content to be modified. Okay, you can make it like a constant, for example. So some constants can be. I can make a random constant like my increase in pay. Okay, so it's a constant that everyone's pay is increase is going to be something like 2% or something like that. So integer increase. This is a convention, okay, if there's a final keyword, make sure it's all capital is equal to 2% or something. Just add a final keyword over here. This will make sure that you are not able to modify this variable. Okay, so now this increase. The variable cannot be modified. Okay, it cannot be modified. Let's look an example of this. Let's look an example of this. So here if I have something like a final keyword, I can say something like this. I can say final int bonus or something is equal to two. Then when I say bonus is equal to, let's say I try to change it to three. It's giving me an error. You can't modify it, you can't modify it. That's basically what it is. Now one more thing I want to show you is that, okay, I already mentioned about like the convention and stuff, but what if we have like a final field inside when we're talking about, you know, the, the classes and stuff. So what if we have a final over there? Let's see, makes sense like to have declaring the class as final. I will teach you that in Inheritance program because that is the concept that it's related to. But here if I talk about I create a class, I create a class A and it just has a final int number is equal to something or I don't have it, or let's say whatever. So it's giving me an error, obviously because it's saying that it is not initialized and final variables have to be initialized. Why? Let's see. Because here you can see that you can't modify it, you can't add it or change it or whatever. That is why you. You should always initialize the final variable when it is declared. This is an important point. Okay, so always, since you can't modify it or whatever, that's the common sense. Always initialize it while declaring it. Always initialize while declaring it. Okay, so that is sort of like the reason, because you can't really change it or whatever. Okay, but unfortunately final guarantees that this immutability that you cannot change it or whatever only when the instance variables are primitive data types and not the reference types of objects and stuff. Why? Because if the instance variable of a reference type has a final modifier behind it, the value of that instance variable will never change. Okay? The reference to the object will never change. It will always refer to the same object, but the value of the object can change. What do I mean by that? Let's see. So what do I mean by that is that this immutability that you cannot change the value. Ok? This is immutable. I'll mention this in the notes as well. This immutability that you cannot change the value is only holding true for primitive data types. Then what if the object itself is final? Let's say you say final student Kunal is equal to new student. Here you can do kunal.name. if you want to change, you can do it to new name. But now you'll be like hey Kunal, you said that final means you cannot make any changes. Then how are you making a change over here? Like this? No, it only means that you cannot make the change in the value when it is primitive data type. But if it is not primitive data type, then you can make the change in the value, but you cannot reassign it. You can't do something like this. Kunal is equal to other object or something like that. I'm not mentioning over here because I'll mention it in my notes. Okay, so that is basically what it means. Let's try to look at an example for this as well. So if I try to look at an example for this, as you can see, initialization is important. But if I try to look an example for this, let's say I take an example over here, I say something like, you know, I say something like string name is equal to nothing. Right? Now in Intellij, you can also create an object like constructor, like alt, enter something like this alt enter constructor name automatically created a constructor for me. Okay, no problem. So I can say something like. Not this. I want to say something like. Let's say student Kunal is equal to new student. And I'm going to pass Kunal Kushwaha to it. Kunal. Okay, one more thing I want to share is. Okay, it should be inside public static void. Main what am I doing? Should be inside public static void. Okay, so here not students. Sorry. A Kunal is equal to new A. A Kunal is equal to new a. Something like that. Okay, now if I just say if I make it as final. Basically I can do like this Kunal name is equal to other name or something. But I can't do something like this. Kunal is equal to new a new object. I can't do it. So when a non primitive. So I will write it down. When a non primitive is final, you cannot reassign it. You can change the values or whatever. This is allowed. But this is not allowed. You cannot reassign it. It will point to the same object with that object. You do what you want, but you can't reassign it. That is sort of like the final keyword with this thing. Okay, I also talked about the final keywords over here and outside it as well. Okay, let's move forward. So we have already talked about garbage collection when a particular object in the heap. So let's talk about like garbage collection and how it works over here in object oriented programming. Obviously we'll cover this in detail when we do it like later on. So if you have something like this and this A is pointing to some object. Now let's say A is pointing to some other object. Please watch the introduction to programming video in that we actually covered this in detail. So this will be removed by garbage collector when when it wants to hit it. Okay, so. So it had to happens in Java automatically. The automatic garbage collection happens. But sometimes when the object is like destroyed with garbage collection, water and whatever you need to perform some sort of an action to handle such situations, Java provides something known as finalization. So we talked about constructor in C. You might have heard about destructors there you free the memory, you create, remove all the objects and stuff manually. In Java it does it automatically. But it Java does give you a way by performing some actions whenever your object is being destroyed. So for example, if a file object is being destroyed, you might want to close the stream or something. Okay. Whenever some object is destroyed, you might want to display a message or something. Okay, you can't do it manually, but what you can do manually is what you can specify what to do when the object is destroyed manually, but you cannot destroy the object manually. You will tell Java, hey, I know you will not allow me to destroy the object manually, so can you please do these things? Whenever you decide to destroy the object, Java will be like, okay, you can specify these specific actions that will occur when the object is about to be, you know, taken away by the garbage collector using finalizer. And a finalizer you can add to a class using the finalize method. Similarly you have constructor. This is known as sort of like a destructor. But it will be called automatically when Java. Basically finalize method will be called when Java is doing garbage collection. Okay, that is how it goes. So right before the object is freed from the memory, Java is going to call the garbage collector and the finalize method. Okay, this is how it. So here what will happen is you will have finalize, finalize like this, something like this. Here it says that it's, it's deprecated over here. But this is basically what it does. So if I just, do, you know, something like something like this, I can say something like object is destroyed or something like that. Okay, object is destroyed, something like that. Here it says that it's actually like deprecating. So might have to look into like the docs or something. But that is basically what Finalize used to do. They might have something different for it. But what is this override thing? What is this throws and throwable? That will be covered later on. For example, if you have this finalize method inside the class A. Similarly you can have different finalize methods in different classes. So whenever the object of class A, any object of class A is freed from the memory, this will be called. Let's try it out. Okay, so let's try to see. Let's say we do, obviously if you put a, you know, print statement over here, like object is being created. You can do something like this object created. But we only want to see when the object is being destroyed. So let me call a few like, Let me call like something like this. I'm going to say let's say a object. Okay, I'm going to create, I'm going to like create some thousand or something objects like this, a lot of objects like this. Okay, and then I'm just going to say object is equal to every time it's creating a new object, starting some random name over here. So you can see once it will be created, then it will be created again. And you can see it's only using the same reference variable. Hence so many objects are being created. And I know that no more than one object can point to the same reference variable. Hence it will be happening something like this internally. Internally it will have let's say obj in the stack memory. In the heap memory an object will be created. It will be pointing to that. Then when the for loop runs again it will create another object. It will be pointing to that. For loop runs again it will create another object. This will happen something like thousands and thousands of times. How many times I have specified in my and this will be removed by garbage collector. Okay, let's try to try to run this thing. Okay. For less number of ones it's not going to I believe call garbage collector. OK let's see. Let's run this program was running already. Let's run this again. Okay, so it says that nothing was shown. Okay, if you want to run this again you can see literally it was not called. So basically all these 10,000 objects that were created, it was not too much load on the memory. So it was like okay, I'm not going to delete any. Let's put some load on the memory like this. Let's create a lot more now. It's definitely going to clear the memory with garbage collector. Yeah. Object is destroyed. Object is destroyed. Okay, so it is destroying the objects. So this object destroying thing, whatever thing happens, it will do it on its own. I can't really say something like this. I can't do something like object finalize. Okay, we can't do it manually. Okay, it's saying like you, you can't do it. You. You can't do it manually. It's raising an exception and we'll cover exception handling as well later on. Okay, don't worry about this. Only new thing in this lecture is this thing and this thing that I have not covered. Override and throwable. So these keywords I will cover later on in future videos. But that's basically what I meant by garbage collection stuff. So you can't really free the memory yourself but you can tell it what to do when the memory is freed. Do check out the new docs for Java like how it deals with that. But it's still like in it's working over here as well. But it's important it is asked in interviews and stuff sometimes like how it deals with garbage collection. That's why I have to tell you. Okay, that was it about this. Just an introductory video in the next few videos we'll talk about like packages, static non static singleton class. We'll talk about properties, polymorphism, overriding, constructor and all these other things. Special methods, inheritance types, encapsulation, abstraction, access control, object class, object cleaning interface, abstract. I'm reading like the syllabus that I have. Abstract classes, interfaces, generics, exception handling, collection framework, lambda expression, enums, file handling and fast input output. All of these things will be covered okay in detail. So please make sure like share and subscribe and one more thing I want you to do is comment. Commenting is important and liking is important and share about it on socials like object bombing started and really excited for the object playlist. Please share about it on social use the hashtag DSA with Kunal Tag me that Community classroom share it on LinkedIn and Twitter. We will read to it and sort of like the learning public initiated. So show your excitement about the object or bombing playlist and let's you know bring this course to as many people as possible. I will teach you literally everything that I just mentioned and it's going to be great because object is very very very very very important for interviews. Make sure you do all these things that I just mentioned. I will see you in the next one. Please make sure you comment down below as well and like share and subscribe and share on socials and yeah, see you in the next one. -"

@app.post("/chapter/{session_id}")
async def get_chapters(session_id: str):
    """Get chapter-wise summary using the transcript"""
    session = video_sessions.get(session_id)
    # if not session:
    #     raise HTTPException(status_code=404, detail="Session not found")

    # if not session.transcript_text:
    #     raise HTTPException(status_code=400, detail="Transcript not yet available")

    from openai import OpenAI
    import json
    import re

    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompt = f"""
You are a helpful AI assistant.

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

        # If markdown formatting like ```json is included
        if raw_output.startswith("```"):
            raw_output = re.sub(r"^```(?:json)?", "", raw_output)
            raw_output = raw_output.rstrip("```").strip()

        chapters = json.loads(raw_output)

        return {
            "session_id": session_id,
            "chapters": chapters
        }

    except Exception as e:
        logger.error(f"Chapter generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate chapters")


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
