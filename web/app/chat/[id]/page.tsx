'use client';
import React, { useEffect, useState, useRef } from 'react';

// Import the actual react-youtube library
import YouTube from 'react-youtube';

// This is the wrapper component, now enhanced with a real YouTube player.
// It simulates the ReactPlayer API for consistency.
const UniversalPlayer = React.forwardRef(({ url, controls, playing, width, height, onError, onReady, onPlay, onPause }:any, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const youtubePlayerRef = useRef<any>(null); // Ref for the actual YouTube player instance

  const [error, setError] = useState<string | null>(null);
  
  // Expose a unified seekTo method through the parent ref
  React.useImperativeHandle(ref, () => ({
    seekTo: (seconds: number) => {
      // Check if the current URL is for YouTube
      if ((url.includes('youtube.com') || url.includes('youtu.be')) && youtubePlayerRef.current) {
        // Use the official YouTube Player API method
        youtubePlayerRef.current.seekTo(seconds, true);
        console.log(`YouTube seek command sent for ${seconds}s`);
      } else if (videoRef.current) {
        // Use the native HTML video API for S3/direct files
        videoRef.current.currentTime = seconds;
        console.log(`HTML5 video seek command sent for ${seconds}s`);
      } else {
        console.error("Player not ready or ref not available for seeking.");
      }
    },
  }));
  
  // Reset error state when URL changes
  useEffect(() => {
    setError(null);
  }, [url]);

  // Determine which player to render
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
              // Save the player instance to our ref
              youtubePlayerRef.current = event.target;
              if (onReady) onReady();
            }}
            onError={(e) => {
              console.error('YouTube Player Error:', e);
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

  // Fallback to the native HTML video player for S3 and other direct links
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


export default function ChatPage() {
  const [videoUrl, setVideoUrl] = useState('');
  const [playerState, setPlayerState] = useState({
    ready: false,
    playing: false,
    error: null
  });
  const playerRef = useRef(null);

  useEffect(() => {
    // Start with a known working video
    setVideoUrl('https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4');
  }, []);

  const handleSeek = (seconds: number) => {
    console.log(`Attempting to seek to ${seconds} seconds`);
    if (playerRef.current && (playerRef.current as any).seekTo) {
      (playerRef.current as any).seekTo(seconds);
      console.log(`Seek command sent to player`);
    } else {
      console.error('Player ref or seekTo method not available');
    }
  };

  const playS3Video = () => {
    setVideoUrl('https://decentralized-web2-quickpay.s3.ap-south-1.amazonaws.com/8e766a0c-b818-4047-8ec2-6bd7eef2e76f.mp4');
    setPlayerState(prev => ({ ...prev, error: null, ready: false }));
  };

  const playYouTubeVideo = () => {
    setVideoUrl('https://www.youtube.com/watch?v=LXb3EKWsInQ');
    setPlayerState(prev => ({ ...prev, error: null, ready: false }));
  };

  const playTestVideo = () => {
    setVideoUrl('https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4');
    setPlayerState(prev => ({ ...prev, error: null, ready: false }));
  };

  return (
    <div className="grid md:grid-cols-2 h-screen bg-[#0f0f0f] text-white">
      {/* Chat side */}
      <div className="h-screen col-span-1 bg-[#1a1a1a] flex flex-col">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Debug info */}
          <div className="bg-[#2a2a2a] p-3 rounded-lg">
            <h3 className="font-bold text-yellow-400 mb-2">Debug Info:</h3>
            <p className="text-sm truncate">URL: {videoUrl || 'None'}</p>
            <p className="text-sm">Player Ready: {playerState.ready ? 'Yes' : 'No'}</p>
            <p className="text-sm">Playing: {playerState.playing ? 'Yes' : 'No'}</p>
            <p className="text-sm">Video Type: {
              videoUrl.includes('youtube') ? 'YouTube (react-youtube)' : 
              videoUrl.includes('.mp4') ? 'Direct MP4 (HTML5 <video>)' : 'Unknown'
            }</p>
            {playerState.error && (
              <p className="text-sm text-red-400">Error: {playerState.error}</p>
            )}
          </div>

          {/* Controls */}
          <div className="space-y-2 pt-4">
            <h3 className="text-lg font-semibold">Player Controls</h3>
            <div className="flex gap-2 flex-wrap">
              <button onClick={playTestVideo} className="bg-purple-600 hover:bg-purple-700 text-white px-3 py-2 rounded-lg text-sm">Test Video</button>
              <button onClick={playS3Video} className="bg-orange-600 hover:bg-orange-700 text-white px-3 py-2 rounded-lg text-sm">Your S3 Video</button>
              <button onClick={playYouTubeVideo} className="bg-red-600 hover:bg-red-700 text-white px-3 py-2 rounded-lg text-sm">YouTube Test</button>
            </div>
            <div className="flex gap-2">
              <button onClick={() => handleSeek(15)} className="bg-green-600 hover:bg-green-700 text-white px-3 py-2 rounded-lg text-sm">Jump to 15s</button>
              <button onClick={() => handleSeek(45)} className="bg-green-600 hover:bg-green-700 text-white px-3 py-2 rounded-lg text-sm">Jump to 45s</button>
            </div>
          </div>
        </div>

        {/* Input box */}
        <div className="bg-[#1e1e1e] p-4 border-t border-gray-700">
          <div className="flex items-center gap-2">
            <textarea
              className="flex-1 p-2 border border-[#444] bg-[#121212] text-white rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
              rows={2}
              placeholder="Type your message..."
            />
            <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors">Send</button>
          </div>
        </div>
      </div>

      {/* Video player side */}
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
  );
}
