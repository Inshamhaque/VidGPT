'use client';
import { BACKEND_URL } from '@/app/config';
import axios from 'axios';
import React, { useEffect, useState, useRef, use } from 'react';
import Navbar from '@/app/components/Navbar';
import YouTube from 'react-youtube';
import { useRouter } from 'next/navigation';
import { ArrowBigRightIcon, ArrowRight, Send } from 'lucide-react';

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

  const isYouTube = url && (url.includes('youtube.com') || url.includes('youtu.be'));
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

interface Chapter {
  title: string;
  start_time: string;
  description: string;
}

interface TranscriptSegment {
  text: string;
  start_time: number;
  end_time: number;
}

interface TranscriptResponse {
  transcript?: TranscriptSegment[];
  language?: string;
  error?: string;
  message?: string;
}

// Transcript Modal Component
const TranscriptModal = ({ isOpen, onClose, transcript, isLoading, onDownload, sessionId, language, error }: {
  isOpen: boolean;
  onClose: () => void;
  transcript: string
  isLoading: boolean;
  onDownload: () => void;
  sessionId: string;
  language?: string;
  error?: string;
}) => {
  if (!isOpen) return null;

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getLanguageFlag = (lang?: string) => {
    if (!lang) return '🌐';
    const langMap: { [key: string]: string } = {
      'en': '🇺🇸', 'es': '🇪🇸', 'fr': '🇫🇷', 'de': '🇩🇪', 'it': '🇮🇹',
      'pt': '🇵🇹', 'ru': '🇷🇺', 'ja': '🇯🇵', 'ko': '🇰🇷', 'zh': '🇨🇳',
      'ar': '🇸🇦', 'hi': '🇮🇳', 'tr': '🇹🇷', 'pl': '🇵🇱', 'nl': '🇳🇱'
    };
    return langMap[lang.toLowerCase()] || '🌐';
  };

//   const transcriptList = transcript?.slice(10)

  const getLanguageName = (lang?: string) => {
    if (!lang) return 'Unknown';
    const langNames: { [key: string]: string } = {
      'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
      'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
      'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
      'tr': 'Turkish', 'pl': 'Polish', 'nl': 'Dutch'
    };
    return langNames[lang.toLowerCase()] || lang.toUpperCase();
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-[#1a1a1a] rounded-xl border border-gray-700 max-w-4xl w-full max-h-[80vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Video Transcript</h2>
              {language && (
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-lg">{getLanguageFlag(language)}</span>
                  <span className="text-sm text-gray-400">{getLanguageName(language)}</span>
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center gap-3">
            {!error && (
              <button
                onClick={onDownload}
                disabled={!transcript || isLoading}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Download
              </button>
            )}
            <button
              onClick={onClose}
              className="w-8 h-8 flex items-center justify-center hover:bg-gray-700 rounded-lg transition-colors text-gray-400 hover:text-white"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-12">
              <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
              <p className="text-gray-400">Loading transcript...</p>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="w-16 h-16 bg-red-500/10 rounded-full flex items-center justify-center mb-4">
                <svg className="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.664-.833-2.464 0L4.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-red-400 mb-2">Transcript Unavailable</h3>
              <p className="text-gray-400 max-w-md leading-relaxed mb-4">{error}</p>
              <div className="bg-gray-800/50 rounded-lg p-4 max-w-md">
                <p className="text-sm text-gray-300 mb-2">
                  <strong>Possible reasons:</strong>
                </p>
                <ul className="text-sm text-gray-400 space-y-1 list-disc list-inside">
                  <li>Video is in a language not yet supported</li>
                  <li>Audio quality is too low for transcription</li>
                  <li>Video contains mostly music or non-speech content</li>
                  <li>Transcription service is temporarily unavailable</li>
                </ul>
              </div>
            </div>
          ) : transcript && transcript.length > 0 ? (
            <div className="space-y-4">
              {transcript}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-gray-400">
              <svg className="w-16 h-16 mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="text-lg">No transcript available</p>
              <p className="text-sm mt-1">The transcript could not be loaded for this video.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

export default function ChatPage({ params }: PageProps) {
  const [videoUrl, setVideoUrl] = useState<null|string>(null);
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
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [isLoadingChapters, setIsLoadingChapters] = useState(false);
  const [showChapters, setShowChapters] = useState(false);
  const [chaptersMinimized, setChaptersMinimized] = useState(false);
  
  // Transcript states
  const [showTranscript, setShowTranscript] = useState(false);
  const [transcript, setTranscript] = useState<string|null>(null);
  const [isLoadingTranscript, setIsLoadingTranscript] = useState(false);
  const [transcriptLanguage, setTranscriptLanguage] = useState<string | undefined>(undefined);
  const [transcriptError, setTranscriptError] = useState<string | undefined>(undefined);

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

  // Fetch transcript
  const fetchTranscript = async () => {
    setIsLoadingTranscript(true);
    setTranscriptError(undefined);
    try {
      const response = await axios.get(`${BACKEND_URL}/transcript/${id}`);
      
        setTranscript(response.data.transcript)
    //   if (data.error || data.message) {
    //     // Handle API errors
    //     setTranscriptError(data.error || data.message || 'Failed to load transcript');
    //     setTranscript(null);
    //     setTranscriptLanguage(undefined);
    //   } else if (data.transcript && data.transcript.length > 0) {
    //     // Successfully loaded transcript
    //     setTranscript(data.transcript);
    //     setTranscriptLanguage(data.language);
    //     setTranscriptError(undefined);
    //   } else {
    //     // Empty or invalid transcript
    //     setTranscriptError('No transcript content available for this video');
    //     setTranscript(null);
    //     setTranscriptLanguage(undefined);
    //   }
    } catch (err: any) {
      console.error("Transcript fetch error:", err);
      
      // Handle different types of errors
      if (err.response?.status === 404) {
        setTranscriptError('Transcript not found for this video');
      } else if (err.response?.status === 422) {
        setTranscriptError('Video language not supported for transcription');
      } else if (err.response?.status === 500) {
        setTranscriptError('Transcription service temporarily unavailable');
      } else if (err.response?.data?.detail) {
        setTranscriptError(err.response.data.detail);
      } else if (err.response?.data?.message) {
        setTranscriptError(err.response.data.message);
      } else {
        setTranscriptError('Unable to load transcript. Please try again later.');
      }
      
      setTranscript(null);
      setTranscriptLanguage(undefined);
    } finally {
      setIsLoadingTranscript(false);
    }
  };

  // Download transcript
  const downloadTranscript = () => {
    if (!transcript || transcript.length === 0) return;

    const formatTime = (seconds: number): string => {
      const mins = Math.floor(seconds / 60);
      const secs = Math.floor(seconds % 60);
      return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const header = `Video Transcript - Session ${id}\n${transcriptLanguage ? `Language: ${transcriptLanguage.toUpperCase()}\n` : ''}Generated: ${new Date().toLocaleString()}\n${'='.repeat(50)}\n\n`;
    
    const transcriptText = transcript
      
    const fullContent = header + transcriptText;

    const blob = new Blob([fullContent], { type: 'text/plain; charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcript_${id}${transcriptLanguage ? `_${transcriptLanguage}` : ''}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Chapter helper 
  const generateChapters = async () => {
    setIsLoadingChapters(true);
    try {
      const res = await axios.post(`${BACKEND_URL}/chapter/${id}`);
      setChapters(res.data.chapters || []);
      setShowChapters(true);
    } catch (err) {
      console.error("Chapter generation error:", err);
      alert("Failed to generate chapters.");
    } finally {
      setIsLoadingChapters(false);
    }
  };

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

  const timeToSeconds = (time: string): number => {
    const [mins, secs] = time.split(':').map(Number);
    return mins * 60 + secs;
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

  const handleShowTranscript = () => {
    setShowTranscript(true);
    if (!transcript && !transcriptError) {
      fetchTranscript();
    }
  };

  return (
    <div className="bg-[#0f0f0f] text-white h-screen flex flex-col">
      <Navbar />
      <div className="grid lg:grid-cols-2 flex-1 overflow-hidden">
        {/* Chat Section */}
        <div className="h-full bg-gradient-to-b from-[#1a1a1a] to-[#0f0f0f] flex flex-col overflow-hidden border-r border-gray-800">
          <div className='flex-1 overflow-y-auto p-6 space-y-6'>
            {messages.length === 0 ? (
              <div className='h-full flex flex-col items-center justify-center text-center'>
                <div className="w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6">
                  <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                </div>
                <h1 className='text-2xl font-bold text-gray-300 mb-2'>
                  Start Your Conversation
                </h1>
                <p className='text-gray-500 max-w-md'>
                  Ask questions about the video, get insights, and explore key moments through our AI-powered chat.
                </p>
              </div>
            ) : (
              <>
                {messages.map((message) => (
                  <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[85%] rounded-2xl p-4 shadow-lg ${
                      message.type === 'user' 
                        ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white' 
                        : 'bg-[#2a2a2a] text-gray-100 border border-gray-700'
                    }`}>
                      {message.type === 'user' ? (
                        <p className="text-sm leading-relaxed">{message.content}</p>
                      ) : (
                        <div className="space-y-4">
                          <p className="text-sm leading-relaxed">{message.answer}</p>
                          {message.timestamps?.length > 0 && (
                            <div className="border-t border-gray-600 pt-4">
                              <div className="flex items-center gap-2 mb-3">
                                <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                <p className="text-xs text-gray-400 font-medium">Relevant timestamps:</p>
                              </div>
                              <div className="space-y-2">
                                {message.timestamps.map((timestamp, index) => (
                                  <div
                                    key={index}
                                    className="bg-[#1e1e1e] rounded-lg p-3 cursor-pointer hover:bg-[#333] transition-all duration-200 border border-gray-700 hover:border-blue-500"
                                    onClick={() => handleSeek(timestamp.start_time)}
                                  >
                                    <div className="flex items-center justify-between mb-2">
                                      <span className="text-xs text-blue-400 font-mono bg-blue-400/10 px-2 py-1 rounded">
                                        {formatTime(timestamp.start_time)} - {formatTime(timestamp.end_time)}
                                      </span>
                                      <span className="text-xs text-gray-500 flex items-center gap-1">
                                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h.01M12 5v.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                        Click to jump
                                      </span>
                                    </div>
                                    <p className="text-xs text-gray-300 italic leading-relaxed">"{timestamp.phrase}"</p>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                          {message.quotes?.length > 0 && (
                            <div className="border-t border-gray-600 pt-4">
                              <div className="flex items-center gap-2 mb-3">
                                <svg className="w-4 h-4 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
                                </svg>
                                <p className="text-xs text-gray-400 font-medium">Key quotes:</p>
                              </div>
                              <div className="space-y-2">
                                {message.quotes.map((quote, index) => (
                                  <div key={index} className="bg-yellow-400/5 border-l-2 border-yellow-400 pl-3 py-2">
                                    <p className="text-xs text-gray-300 italic leading-relaxed">"{quote}"</p>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                      <div className="text-xs text-gray-400 mt-3 opacity-70">
                        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-[#2a2a2a] border border-gray-700 rounded-2xl p-4 shadow-lg">
                      <div className="flex items-center space-x-3">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        </div>
                        <span className="text-xs text-gray-400">AI is analyzing...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </>
            )}
          </div>
          
          {/* Enhanced Input Section */}
          <div className="bg-[#1e1e1e] p-6 border-t border-gray-700">
            <div className="flex items-end gap-3">
              <div className="flex-1">
                <textarea
                  className="w-full p-4 border border-gray-600 bg-[#121212] text-white rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all placeholder-gray-500"
                  rows={2}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about the video content..."
                  disabled={isLoading}
                />
                
                {isRecording && (
                  <div className="mt-3 flex items-center gap-2 text-red-400 text-sm bg-red-400/10 px-3 py-2 rounded-lg">
                    <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                    Recording... Click stop when finished
                  </div>
                )}
              </div>
              
              <div className="flex flex-col gap-2">
                {!isRecording ? (
                  <button
                    onClick={startRecording}
                    disabled={isLoading}
                    className="bg-slate-600 hover:bg-slate-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white p-3 rounded-xl transition-all text-sm flex items-center justify-center min-w-[50px]"
                    title="Start voice recording"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                  </button>
                ) : (
                  <button
                    onClick={stopRecording}
                    className="bg-gray-700 hover:bg-gray-600 text-white p-3 rounded-xl transition-all text-sm flex items-center justify-center min-w-[50px] animate-pulse"
                    title="Stop recording"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10h6v4H9z" />
                    </svg>
                  </button>
                )}
                
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || isLoading}
                  className="bg-gradient-to-r from-gray-600 to-gray-700 hover:from-blue-700 hover:to-blue-800 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed text-white p-3 rounded-xl transition-all font-medium min-w-[50px] flex items-center justify-center"
                >
                  {isLoading ? (
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  ) : (
                    <Send />
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Video Section */}
        <div className="bg-black flex flex-col relative">
          {/* Video Player */}
          <div className="flex-1 flex items-center justify-center p-6">
            <div className="w-full max-w-5xl aspect-video bg-black rounded-xl overflow-hidden shadow-2xl border border-gray-800">
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

          {/* Action Buttons */}
          <div className="p-6 bg-gradient-to-t from-black/50 to-transparent">
            <div className="flex flex-wrap gap-3 justify-center">
              {!chapters && <button
                onClick={generateChapters}
                disabled={isLoadingChapters || chapters.length>0}
                className="flex items-center gap-2 bg-gradient-to-r from-gray-500 to-gray-500 hover:from-gray-600 hover:to-gray-600 disabled:from-gray-600 disabled:to-gray-600 hover:disabled:cursor-not-allowed text-white font-semibold px-6 py-3 rounded-xl shadow-lg transition-all transform hover:scale-105 disabled:scale-100"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                {isLoadingChapters ? "Generating..." : "Generate Chapters"}
              </button>}

              <button
                onClick={handleShowTranscript}
                className="flex items-center gap-2 bg-gradient-to-r from-gray-500 to-gray-500 hover:from-gray-600 hover:to-gray-600 text-white font-semibold px-6 py-3 rounded-xl shadow-lg transition-all transform hover:scale-105"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                View Transcript
              </button>

              
              
            </div>
          </div>

          {/* Chapters Panel */}
          {showChapters && chapters.length > 0 && (
            <div className={`absolute top-6 right-6 max-w-sm w-full bg-[#1a1a1a]/95 backdrop-blur-sm rounded-xl border border-gray-700 shadow-2xl transition-all duration-300 ${
              chaptersMinimized ? 'max-h-16' : 'max-h-96'
            }`}>
              <div className="p-4 border-b border-gray-700 flex items-center justify-between">
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <svg className="w-5 h-5 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                  Chapters
                  <span className="text-xs bg-yellow-400/20 text-yellow-300 px-2 py-0.5 rounded-full">
                    {chapters.length}
                  </span>
                </h3>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setChaptersMinimized(!chaptersMinimized)}
                    className="text-gray-400 hover:text-white transition-colors p-1 rounded hover:bg-gray-700"
                    title={chaptersMinimized ? "Expand chapters" : "Minimize chapters"}
                  >
                    <svg 
                      className={`w-4 h-4 transition-transform duration-200 ${chaptersMinimized ? 'rotate-180' : ''}`} 
                      fill="none" 
                      stroke="currentColor" 
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                  <button
                    onClick={() => setShowChapters(false)}
                    className="text-gray-400 hover:text-white transition-colors p-1 rounded hover:bg-gray-700"
                    title="Close chapters"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>
              {!chaptersMinimized && (
                <div className="p-2 space-y-2 overflow-y-auto" style={{ maxHeight: 'calc(24rem - 4rem)' }}>
                  {chapters.map((chapter, idx) => (
                    <div
                      key={idx}
                      className="cursor-pointer p-3 bg-[#2a2a2a] hover:bg-[#333] rounded-lg transition-all border border-gray-700 hover:border-yellow-400"
                      onClick={() => handleSeek(timeToSeconds(chapter?.start_time))}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <h4 className="text-sm font-semibold text-white line-clamp-2">{chapter.title}</h4>
                        <span className="text-xs text-yellow-400 font-mono bg-yellow-400/10 px-2 py-0.5 rounded ml-2 min-w-fit">
                          {chapter.start_time}
                        </span>
                      </div>
                      <p className="text-xs text-gray-300 line-clamp-2">{chapter.description}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Transcript Modal */}
      <TranscriptModal
        isOpen={showTranscript}
        onClose={() => setShowTranscript(false)}
        transcript={transcript}
        isLoading={isLoadingTranscript}
        onDownload={downloadTranscript}
        sessionId={id}
      />
    </div>
  );
}