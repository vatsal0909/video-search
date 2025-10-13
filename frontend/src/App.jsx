import React, { useState } from 'react';
import Header from './components/Header';
import SearchBar from './components/SearchBar';
import ResultsGrid from './components/ResultsGrid';
import VideoPlayer from './components/VideoPlayer';
import VideoUpload from './components/VideoUpload';
import { searchClips } from './services/api';
import { AlertCircle, Sparkles } from 'lucide-react';

function App() {
  const [currentPage, setCurrentPage] = useState('search'); // 'search' or 'upload'
  const [clips, setClips] = useState([]);
  const [total, setTotal] = useState(0);
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedClip, setSelectedClip] = useState(null);

  const handle_search = async (searchQuery) => {
    setIsLoading(true);
    setError(null);
    setQuery(searchQuery);

    try {
      const response = await searchClips(searchQuery, 10);
      setClips(response.clips);
      setTotal(response.total);
    } catch (err) {
      setError('Failed to search videos. Please try again.');
      console.error('Search error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handle_clip_click = (clip) => {
    setSelectedClip(clip);
  };

  const close_player = () => {
    setSelectedClip(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50">
      <Header currentPage={currentPage} onPageChange={setCurrentPage} />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentPage === 'upload' ? (
          <VideoUpload />
        ) : (
          <>
        {/* Hero Section */}
        <div className="text-center mb-12 pt-8">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary-100 text-primary-700 rounded-full text-sm font-medium mb-4">
            <Sparkles size={16} />
            Powered by TwelveLabs & AWS
          </div>
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            Search Videos
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Find exact moments in your videos using natural language.
          </p>
        </div>

        {/* Search Bar */}
        <div className="mb-12">
          <SearchBar onSearch={handle_search} isLoading={isLoading} />
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-8 p-4 bg-red-50 border border-red-200 rounded-xl flex items-center gap-3 text-red-700 animate-fade-in">
            <AlertCircle size={20} />
            <span>{error}</span>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="text-center py-16">
            <div className="inline-block w-16 h-16 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin mb-4"></div>
            <p className="text-gray-600">Searching videos...</p>
          </div>
        )}

        {/* Results */}
        {!isLoading && (clips.length > 0 || query) && (
          <ResultsGrid
            clips={clips}
            total={total}
            query={query}
            onClipClick={handle_clip_click}
          />
        )}

        {/* Empty State */}
        {!isLoading && !query && clips.length === 0 && (
          <div className="text-center">
            <div className="max-w-md mx-auto">
              <div className="w-24 h-24 bg-gradient-to-br from-primary-100 to-purple-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <Sparkles size={48} className="text-primary-600" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-3">
                Start Your Search
              </h3>
              <p className="text-gray-600 mb-6">
                Enter a description of what you're looking for in the search bar above.
                Try queries like "person walking" or "sunset scene".
              </p>
              <div className="flex flex-wrap gap-2 justify-center">
                <button
                  onClick={() => handle_search('person walking in park')}
                  className="px-4 py-2 bg-white border border-gray-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-all text-sm"
                >
                  person walking in park
                </button>
                <button
                  onClick={() => handle_search('sunset scene')}
                  className="px-4 py-2 bg-white border border-gray-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-all text-sm"
                >
                  sunset scene
                </button>
                <button
                  onClick={() => handle_search('people talking')}
                  className="px-4 py-2 bg-white border border-gray-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-all text-sm"
                >
                  people talking
                </button>
              </div>
            </div>
          </div>
        )}
        </>
        )}
      </main>

      {/* Video Player Modal */}
      {selectedClip && (
        <VideoPlayer clip={selectedClip} onClose={close_player} />
      )}
    </div>
  );
}

export default App;
