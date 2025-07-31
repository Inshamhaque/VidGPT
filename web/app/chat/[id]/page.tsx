'use client';
import { BACKEND_URL } from '@/app/config';
import axios from 'axios';
import React, { useEffect, useState, useRef, use } from 'react';
import Navbar from '@/app/components/Navbar';
import YouTube from 'react-youtube';
import { useRouter } from 'next/navigation';

const UniversalPlayer = React.forwardRef(({ url, controls, playing, width, height, onError, onReady, onPlay, onPause }: any, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const youtubePlayerRef = useRef<any>(null);
  const [error, setError] = useState<string | null>(null);

  React.useImperativeHandle(ref, () => ({
    seekTo: (seconds: number) => {
      if ((url.includes('youtube.com') || url.includes('youtu.be')) && youtubePlayerRef.current) {
        youtubePlayerRef.current.seekTo(seconds, true);
      } else if (videoRef.current) {
        videoRef.current.currentTime = seconds;
      }
    },
  }));

  useEffect(() => {
    setError(null);
  }, [url]);

  const isYouTube = url.includes('youtube.com') || url.includes('youtu.be');
  const videoId = isYouTube ? url.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)/)?.[1] : null;

  if (isYouTube) {
    return (
      <div style={{ width, height, position: 'relative', backgroundColor: 'black' }}>
        {videoId ? (
          <YouTube
            videoId={videoId}
            opts={{
              height: '100%',
              width: '100%',
              playerVars: {
                autoplay: playing ? 1 : 0,
                controls: controls ? 1 : 0,
              },
            }}
            onReady={(event) => {
              youtubePlayerRef.current = event.target;
              if (onReady) onReady();
            }}
            onError={(e) => {
              setError(`YouTube Error: ${e.data}`);
              if (onError) onError(e);
            }}
            onPlay={onPlay}
            onPause={onPause}
            className="w-full h-full"
          />
        ) : (
          <div className="flex items-center justify-center h-full bg-gray-800 text-white">
            Invalid YouTube URL
          </div>
        )}
      </div>
    );
  }

  return (
    <div style={{ width, height, position: 'relative' }}>
      <video
        ref={videoRef}
        src={url}
        controls={controls}
        width="100%"
        height="100%"
        style={{ backgroundColor: 'black' }}
        autoPlay={playing}
        onLoadedData={() => onReady && onReady()}
        onPlay={() => onPlay && onPlay()}
        onPause={() => onPause && onPause()}
        onError={(e: any) => {
          const errorMsg = `Video error: ${e.target.error?.message || 'Could not load video.'}`;
          setError(errorMsg);
          if (onError) onError(e);
        }}
      >
        Your browser does not support the video tag.
      </video>
      {error && (
        <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center bg-red-900/80 text-white p-4 pointer-events-none">
          <div className="text-center">
            <p className="font-bold">Player Error:</p>
            <p className="text-sm">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
});

interface Message {
  id: string;
  type: 'user' | 'ai';
  content: string;
  answer?: string;
  quotes?: string[];
  timestamps?: Array<{
    phrase: string;
    start_time: number;
    end_time: number;
  }>;
  timestamp: Date;
}

interface PageProps {
  params: {
    id: string;
  };
}

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

export default function ChatPage({ params }: PageProps) {
  const [videoUrl, setVideoUrl] = useState('');
  const [playerState, setPlayerState] = useState({
    ready: false,
    playing: false,
    error: null
  });
  const { id } = use(params);
  const playerRef = useRef(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchVideo = async () => {
      const response = await axios.get(`${BACKEND_URL}/session/${id}`);
      setVideoUrl(response.data.video_url);
    };
    fetchVideo();
  }, [id]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const startRecording = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream);

    mediaRecorderRef.current.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data);
      }
    };

    mediaRecorderRef.current.onstop = async () => {
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
      audioChunksRef.current = [];

      // Stop all tracks to free up microphone
      stream.getTracks().forEach(track => track.stop());

      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');

      try {
        const res = await axios.post(`${BACKEND_URL}/whisper`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        const transcribedText = res.data.transcription;
        setInput(transcribedText);
      } catch (err) {
        console.error('Transcription error:', err);
        alert("Transcription failed. Please try again.");
      } finally {
        setIsRecording(false);
      }
    };

    audioChunksRef.current = [];
    mediaRecorderRef.current.start();
    setIsRecording(true);
    
    // Optional: Auto-stop after 30 seconds (you can adjust or remove this)
    setTimeout(() => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        stopRecording();
      }
    }, 30000);
  } catch (err) {
    alert("Microphone access error. Please check your permissions.");
    console.error(err);
    setIsRecording(false);
  }
};

const stopRecording = () => {
  if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
    mediaRecorderRef.current.stop();
  }
};

  const handleSeek = (seconds: number) => {
    if (playerRef.current && (playerRef.current as any).seekTo) {
      (playerRef.current as any).seekTo(seconds);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await axios.post(`${BACKEND_URL}/chat`, {
        session_id: id,
        message: userMessage.content
      });

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        content: response.data.answer,
        answer: response.data.answer,
        quotes: response.data.quotes,
        timestamps: response.data.timestamps,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch {
      setMessages(prev => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          type: 'ai',
          content: 'Sorry, I encountered an error while processing your request.',
          timestamp: new Date()
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="bg-[#0f0f0f] text-white h-screen flex flex-col">
      <Navbar />
      <div className="grid md:grid-cols-2 flex-1 overflow-hidden">
        <div className="h-full col-span-1 bg-[#1a1a1a] flex flex-col overflow-hidden">
          <div className='flex-1 overflow-y-auto p-4 space-y-4'>
            {messages.length === 0 ? (
              <div className='h-full'>
                <h1 className='text-gray-500 h-full w-full flex justify-center items-center font-mono text-xl'>
                  Start Chatting with the video!
                </h1>
              </div>
            ) : (
              <>
                {messages.map((message) => (
                  <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[80%] rounded-lg p-3 ${message.type === 'user' ? 'bg-blue-600 text-white' : 'bg-[#2a2a2a] text-gray-100'}`}>
                      {message.type === 'user' ? (
                        <p className="text-sm">{message.content}</p>
                      ) : (
                        <div className="space-y-3">
                          <p className="text-sm leading-relaxed">{message.answer}</p>
                          {message.timestamps?.length > 0 && (
                            <div className="border-t border-gray-600 pt-3">
                              <p className="text-xs text-gray-400 mb-2">Relevant timestamps:</p>
                              <div className="space-y-2">
                                {message.timestamps.map((timestamp, index) => (
                                  <div
                                    key={index}
                                    className="bg-[#1e1e1e] rounded p-2 cursor-pointer hover:bg-[#333] transition-colors"
                                    onClick={() => handleSeek(timestamp.start_time)}
                                  >
                                    <div className="flex items-center justify-between mb-1">
                                      <span className="text-xs text-blue-400 font-mono">
                                        {formatTime(timestamp.start_time)} - {formatTime(timestamp.end_time)}
                                      </span>
                                      <span className="text-xs text-gray-500">Click to jump</span>
                                    </div>
                                    <p className="text-xs text-gray-300 italic">"{timestamp.phrase}"</p>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                          {message.quotes?.length > 0 && (
                            <div className="border-t border-gray-600 pt-3">
                              <p className="text-xs text-gray-400 mb-2">Key quotes:</p>
                              <div className="space-y-1">
                                {message.quotes.map((quote, index) => (
                                  <p key={index} className="text-xs text-gray-300 italic">
                                    "{quote}"
                                  </p>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                      <div className="text-xs text-gray-400 mt-2">
                        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-[#2a2a2a] rounded-lg p-3">
                      <div className="flex items-center space-x-2">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        </div>
                        <span className="text-xs text-gray-400">AI is thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </>
            )}
          </div>
          {/* Input + Whisper */}
          <div className="bg-[#1e1e1e] p-4 border-t border-gray-700">
  <div className="flex items-center gap-2">
    <textarea
      className="flex-1 p-2 border border-[#444] bg-[#121212] text-white rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
      rows={2}
      value={input}
      onChange={(e) => setInput(e.target.value)}
      onKeyPress={handleKeyPress}
      placeholder="Type your message or use voice recording..."
      disabled={isLoading}
    />
    
    {/* Recording buttons */}
    <div className="flex gap-1">
      {!isRecording ? (
        <button
          onClick={startRecording}
          disabled={isLoading}
          className="bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-3 py-2 rounded-lg transition-all text-sm flex items-center gap-1"
          title="Start voice recording"
        >
          🎤 Record
        </button>
      ) : (
        <button
          onClick={stopRecording}
          className="bg-gray-700 hover:bg-gray-600 text-white px-3 py-2 rounded-lg transition-all text-sm flex items-center gap-1 animate-pulse"
          title="Stop recording"
        >
          ⏹️ Stop
        </button>
      )}
    </div>
    
    <button
      onClick={handleSend}
      disabled={!input.trim() || isLoading}
      className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg transition-colors"
    >
      {isLoading ? 'Sending...' : 'Send'}
    </button>
  </div>
  
  {/* Recording status indicator */}
  {isRecording && (
    <div className="mt-2 flex items-center gap-2 text-red-400 text-sm">
      <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
      Recording... (Click stop when finished)
    </div>
  )}
</div>
        </div>
        {/* Video Section */}
        <div className="col-span-1 bg-[#000000] flex items-center justify-center p-4">
          <div className="w-full max-w-4xl aspect-video bg-black rounded-lg overflow-hidden">
            <UniversalPlayer
              ref={playerRef}
              url={videoUrl}
              controls={true}
              playing={playerState.playing}
              width="100%"
              height="100%"
              onError={(e: any) => {
                setPlayerState(prev => ({ ...prev, error: e.message || 'An unknown error occurred' }));
              }}
              onReady={() => {
                setPlayerState(prev => ({ ...prev, ready: true }));
              }}
              onPlay={() => {
                setPlayerState(prev => ({ ...prev, playing: true }));
              }}
              onPause={() => {
                setPlayerState(prev => ({ ...prev, playing: false }));
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
