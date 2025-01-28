import React, { useState, useEffect } from 'react';
import { Search, ExternalLink, Mic, MicOff } from 'lucide-react';

function App() {
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [recognition, setRecognition] = useState<any>(null);

  // Initialize voice recognition on component mount
  useEffect(() => {
    // Check if browser supports speech recognition
    const SpeechRecognition = 
      (window as any).SpeechRecognition || 
      (window as any).webkitSpeechRecognition;

    if (SpeechRecognition) {
      const recognitionInstance = new SpeechRecognition();
      recognitionInstance.continuous = false;
      recognitionInstance.lang = 'en-US';

      recognitionInstance.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setQuery(transcript);
        setIsListening(false);
      };

      recognitionInstance.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };

      recognitionInstance.onend = () => {
        setIsListening(false);
      };

      setRecognition(recognitionInstance);
    }
  }, []);

  // Sample data to simulate search results
  const sampleResults = {
    answer: "AI Retail Suite and MicroStrategy cater to different business needs with distinct features and pricing models.\n\nAI Retail Suite is specialized for retail operations, providing tools for retail analytics, dynamic pricing, and predictive capabilities. It offers intuitive dashboards and AI-driven insights that are easy to adopt for retail teams. It also forecasts inventory needs with 95% accuracy, reducing stockouts and overstock.\n\nOn the other hand, MicroStrategy is more comprehensive in its feature set and pricing flexibility. It offers features like HyperIntelligence, which provides instant data insights, Dossier for creating and sharing dynamic analytics reports, and Embedded Analytics for seamless integration of analytics within applications.",
    citations: [
      {
        url: "https://www.spendflo.com/blog/microstrategy-pricing-guide",
        title: "MicroStrategy Pricing Guide 2024",
        source: "Spendflo",
        snippet: "Comprehensive analysis of MicroStrategy's pricing structure."
      },
      {
        url: "https://www.retailtouchpoints.com/features/solution-spotlight/eversight-launches-ai-powered-pricing-suite",
        title: "AI-Powered Retail Analytics Solutions",
        source: "Retail TouchPoints",
        snippet: "Detailed review of AI Retail Suite's analytics capabilities."
      }
    ],
    internal_citations: [
      {
        number: 1,
        source: "Internal Documentation",
        title: "Competitive Analysis Report",
        text: "Comparison of AI Retail Suite with leading competitors in the market."
      }
    ]
  };

  const handleSearch = () => {
    if (!query.trim()) return;

    // Simulate loading
    setIsLoading(true);
    
    // Simulate API delay
    setTimeout(() => {
      setSearchResults(sampleResults);
      setIsLoading(false);
    }, 1500);
  };

  const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      handleSearch();
    }
  };

  const toggleVoiceInput = () => {
    if (!recognition) {
      alert('Speech recognition is not supported in this browser.');
      return;
    }

    if (isListening) {
      recognition.stop();
      setIsListening(false);
    } else {
      try {
        recognition.start();
        setIsListening(true);
      } catch (error) {
        console.error('Error starting speech recognition:', error);
        setIsListening(false);
      }
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <svg className="w-8 h-8 text-white" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path>
              </svg>
              <span className="ml-2 text-2xl font-bold text-white">Datachime</span>
            </div>
            <div className="text-white/90 font-medium">
              {new Date().toLocaleString('en-US', { 
                weekday: 'long', 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
              })}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Search Bar */}
        <div className="max-w-3xl mx-auto mb-8">
          <div className="relative">
            <input
              type="text"
              className="w-full px-4 py-3 pr-20 text-gray-900 placeholder-gray-500 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="Ask anything..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
            />
            <div className="absolute right-0 top-0 h-full flex items-center pr-3 space-x-2">
              {/* Voice Input Button */}
              <button 
                className={`p-2 ${isListening ? 'text-red-600' : 'text-gray-400 hover:text-indigo-600'}`}
                onClick={toggleVoiceInput}
              >
                {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
              </button>

              {/* Search Button */}
              <button 
                className={`p-2 ${query.trim() ? 'text-indigo-600 hover:text-indigo-800' : 'text-gray-400'}`}
                onClick={handleSearch}
                disabled={!query.trim()}
              >
                <Search className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="flex justify-center items-center my-8">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-indigo-500 mr-3"></div>
            <span className="text-gray-600">Searching...</span>
          </div>
        )}

        {/* Search Results */}
        {searchResults && !isLoading && (
          <div className="flex gap-8">
            {/* Main Answer Column */}
            <div className="flex-1 max-w-3xl">
              <div className="bg-white rounded-lg shadow-sm p-6">
                {searchResults.answer.split('\n\n').map((paragraph: string, idx: number) => (
                  <p key={idx} className="mb-4 text-gray-700 leading-relaxed text-lg">
                    {paragraph}
                  </p>
                ))}
              </div>
            </div>

            {/* Citations Column */}
            <div className="w-96 flex-shrink-0">
              <div className="sticky top-4 rounded-xl bg-gradient-to-b from-indigo-50 to-purple-50 p-6">
                <h3 className="text-sm font-semibold text-indigo-900 mb-4">Sources</h3>
                
                {/* External Citations */}
                <div className="space-y-4">
                  {searchResults.citations.map((citation: any, idx: number) => (
                    <div key={idx} className="bg-white/80 backdrop-blur-sm border border-indigo-100 rounded-lg overflow-hidden hover:bg-white transition-colors duration-200">
                      <div className="p-4">
                        <div className="flex items-center justify-between mb-2">
                          <a 
                            href={citation.url} 
                            className="text-indigo-600 hover:text-indigo-800"
                            target="_blank"
                            rel="noopener noreferrer"
                          >{citation.url}
                            <ExternalLink className="w-4 h-4" />
                          </a>
                        </div>
                        
                      </div>
                    </div>
                  ))}
                </div>

                {/* Internal Citations */}
                <h3 className="text-sm font-semibold text-indigo-900 mt-6 mb-4">Internal Sources</h3>
                <div className="space-y-4">
                  {searchResults.internal_citations.map((citation: any, idx: number) => (
                    <div key={idx} className="bg-white/60 backdrop-blur-sm border border-purple-100 rounded-lg p-4 hover:bg-white/80 transition-colors duration-200">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-purple-600">
                          {citation.source}
                        </span>
                        <span className="text-xs font-medium text-purple-500">
                          Citation {citation.number}
                        </span>
                      </div>
                      <h4 className="text-sm font-semibold text-gray-900 mb-2">
                        {citation.title}
                      </h4>
                      <p className="text-sm text-gray-600">
                        {citation.text}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;