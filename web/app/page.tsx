'use client'
import { useState, useEffect } from "react";
import { FileUpload } from "./components/ui/fileupload";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import { MultiStepLoader } from "./components/ui/multi-step-loader";
import { useRouter } from "next/navigation";

export default function Home() {
  const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000";
  console.log(BACKEND_URL)
  const [mode, setMode] = useState<"link" | "upload">("link");
  const [url, setUrl] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const router = useRouter();

  const [currentStep, setCurrentStep] = useState(0);
const steps = [
  { key: 'transcript_generated', text: 'Generating transcript from video...' },
  { key: 'index_created', text: 'Creating searchable index...' },
  { key: 'spoken_words_indexed', text: 'Indexing spoken words for timestamps...' },
  { key: 'processing_complete', text: 'Finalizing processing...' }
];

const getStepIndexFromStatus = (status: any) => {
  if (status.error) return -1;
  if (status.processing_complete) return steps.length;
  return steps.findIndex(step => !status[step.key]);
};

const getVisibleSteps = (status: any) => {
  const idx = getStepIndexFromStatus(status);
  if (idx === -1) return [{ text: 'Error occurred during processing' }];
  return steps.slice(0, idx + 1).map(s => ({ text: s.text }));
};


  // Poll the status endpoint to track processing progress
  useEffect(() => {
    if (!sessionId || !isProcessing) return;

    const pollStatus = async () => {
      try {
        const response = await axios.get(`${BACKEND_URL}/status/${sessionId}`);
        const status = response.data;

        const nextStep = getStepIndexFromStatus(status);
        setCurrentStep(nextStep);
        if (status.processing_complete) {
          setIsProcessing(false);
          toast.success("Video processed successfully! Redirecting to chat...");
          
          // Redirect to chat page after a brief delay
          setTimeout(() => {
            router.push(`/chat/${sessionId}`);
          }, 1500);
        } else if (status.error) {
          setIsProcessing(false);
          toast.error(`Processing failed: ${status.error}`);
          setUrl("");
          setSessionId("");
        }
      } catch (error) {
        console.error("Error polling status:", error);
        // Continue polling even if there's an error
      }
    };

    // Poll every 2 seconds
    const interval = setInterval(pollStatus, 2000);
    
    // Initial poll
    pollStatus();

    return () => clearInterval(interval);
  }, [sessionId, isProcessing, BACKEND_URL]);

  const handleSendLink = async () => {
    if (!url.trim()) {
      toast.error("Please enter a valid YouTube URL");
      return;
    }

    try {
      setIsProcessing(true);
      
      const response = await axios.post(`${BACKEND_URL}/upload-url`, {
        url: url.trim()
      });

      if (response.status === 500) {
        setUrl("");
        setIsProcessing(false);
        return toast.error('Some error occurred', {
          position: "top-right"
        });
      }

      const newSessionId = response.data.session_id;
      setSessionId(newSessionId);
      
      toast.success("Video upload started! Processing...", {
        position: "top-right"
      });

    } catch (error) {
      setIsProcessing(false);
      setUrl("");
      console.error("Upload error:", error);
      
      if (axios.isAxiosError(error)) {
        const errorMessage = error.response?.data?.detail || "Upload failed";
        toast.error(errorMessage, {
          position: "top-right"
        });
      } else {
        toast.error("An unexpected error occurred", {
          position: "top-right"
        });
      }
    }
  };

  const handleFileUpload = async (files: File[]) => {
    if (!files || files.length === 0) return;

    const file = files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
      setIsProcessing(true);
      
      const response = await axios.post(`${BACKEND_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const newSessionId = response.data.session_id;
      setSessionId(newSessionId);
      
      toast.success("File upload started! Processing...", {
        position: "top-right"
      });

    } catch (error) {
      setIsProcessing(false);
      console.error("File upload error:", error);
      
      if (axios.isAxiosError(error)) {
        const errorMessage = error.response?.data?.detail || "File upload failed";
        toast.error(errorMessage, {
          position: "top-right"
        });
      } else {
        toast.error("An unexpected error occurred", {
          position: "top-right"
        });
      }
    }
  };

  return (
    <>
      {/* MultiStepLoader Overlay */}
      <MultiStepLoader
        loadingStates={steps.slice(0, currentStep + 1).map(s => ({ text: s.text }))}
        loading={isProcessing}
        duration={2000}
        loop={false}
      />


      <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black relative overflow-hidden">
        {/* Subtle Background Elements */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-gray-800/20 via-black/40 to-black"></div>
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-gray-600/5 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-gray-500/5 rounded-full blur-3xl"></div>
        
        {/* Navbar */}
        <nav className="relative z-10 border-b border-gray-800 bg-black/80 backdrop-blur-xl px-6 py-4">
          <div className="flex items-center justify-between max-w-6xl mx-auto">
            <h1 className="text-2xl font-bold text-white font-mono tracking-wider">
              VIDGPT
            </h1>
            <button className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-white transition-all duration-300">
              Toggle Theme
            </button>
          </div>
        </nav>

        {/* Hero Section */}
        <div className="relative z-10 flex flex-col items-center justify-center px-6 py-20">
          <div className="text-center max-w-2xl mx-auto mb-12">
            <h2 className="text-4xl md:text-6xl font-bold mb-6 text-white font-mono tracking-tight">
              Chat with any video you want!
            </h2>
            <p className="text-xl text-gray-300 font-mono font-light tracking-wide">
              Upload a video file or paste a YouTube link. Let AI turn it into an interactive experience.
            </p>
          </div>

          {/* Toggle between Link / Upload */}
          <div className="flex bg-gray-900/80 backdrop-blur-sm p-1 rounded-full mb-8 border border-gray-700">
            <button
              onClick={() => setMode("link")}
              className={`px-6 py-3 rounded-full text-sm font-medium transition-all duration-300 ${
                mode === "link"
                  ? "bg-gray-700 text-white shadow-lg"
                  : "text-gray-400 hover:text-white hover:bg-gray-800"
              }`}
              disabled={isProcessing}
            >
              Use Link
            </button>
            <button
              onClick={() => setMode("upload")}
              className={`px-6 py-3 rounded-full text-sm font-medium transition-all duration-300 ${
                mode === "upload"
                  ? "bg-gray-700 text-white shadow-lg"
                  : "text-gray-400 hover:text-white hover:bg-gray-800"
              }`}
              disabled={isProcessing}
            >
              Upload File
            </button>
          </div>

          {/* Conditional Input with Fixed Height Container */}
          <div className="w-full max-w-md">
            <div className="min-h-[140px] flex items-center justify-center">
              {mode === "link" ? (
                <div className="flex flex-col space-y-6 w-full">
                  <div className="relative">
                    <input
                      onChange={(e) => {
                        setUrl(e.target.value);
                      }}
                      value={url}
                      type="url"
                      placeholder="Paste YouTube link here..."
                      className="w-full px-4 py-4 border border-gray-700 rounded-xl bg-gray-900/80 backdrop-blur-sm text-white placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-600 focus:border-transparent transition-all duration-300 hover:bg-gray-900 disabled:opacity-50 disabled:cursor-not-allowed"
                      disabled={isProcessing}
                    />
                  </div>
                  <button 
                    onClick={handleSendLink}
                    disabled={isProcessing || !url.trim()}
                    className="relative group w-full py-4 px-6 bg-gray-800 hover:bg-gray-700 text-white font-semibold rounded-xl transition-all duration-300 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-gray-600 focus:ring-offset-2 focus:ring-offset-black border border-gray-700 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:bg-gray-800"
                  >
                    <span className="relative z-10">
                      {isProcessing ? "Processing..." : "Let's Go 🚀"}
                    </span>
                  </button>
                </div>
              ) : (
                <div className="w-full">
                  <FileUpload 
                    onChange={handleFileUpload}
                    disabled={isProcessing}
                  />
                </div>
              )}
            </div>
          </div>

          {/* Feature Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16 max-w-4xl mx-auto">
            <div className="p-6 rounded-xl bg-gray-900/80 backdrop-blur-sm border border-gray-800 hover:bg-gray-800/80 transition-all duration-300 hover:scale-105">
              <div className="text-3xl mb-3">🎥</div>
              <h3 className="text-lg font-semibold text-white mb-2">Any Video Format</h3>
              <p className="text-gray-400 text-sm">Support for YouTube links and direct file uploads</p>
            </div>
            <div className="p-6 rounded-xl bg-gray-900/80 backdrop-blur-sm border border-gray-800 hover:bg-gray-800/80 transition-all duration-300 hover:scale-105">
              <div className="text-3xl mb-3">🤖</div>
              <h3 className="text-lg font-semibold text-white mb-2">AI-Powered Chat</h3>
              <p className="text-gray-400 text-sm">Ask questions and get intelligent responses about your video</p>
            </div>
            <div className="p-6 rounded-xl bg-gray-900/80 backdrop-blur-sm border border-gray-800 hover:bg-gray-800/80 transition-all duration-300 hover:scale-105">
              <div className="text-3xl mb-3">⚡</div>
              <h3 className="text-lg font-semibold text-white mb-2">Lightning Fast</h3>
              <p className="text-gray-400 text-sm">Quick processing and instant responses</p>
            </div>
          </div>
        </div>
      </div>

      <ToastContainer
        position="top-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="dark"
      />
    </>
  );
}