import VideoClipCard from './VideoClipCard';
import { FileVideo } from 'lucide-react';

const ResultsGrid = ({ clips, total, query, onClipClick }) => {
  if (!clips || clips.length === 0) {
    return (
      <div className="text-center py-16">
        <div className="inline-flex items-center justify-center w-20 h-20 bg-gray-100 rounded-full mb-4">
          <FileVideo size={40} className="text-gray-400" />
        </div>
        <h3 className="text-xl font-semibold text-gray-700 mb-2">No results found</h3>
        <p className="text-gray-500">
          {query ? `No video clips match "${query}"` : 'Try searching for something'}
        </p>
      </div>
    );
  }

  return (
    <div className="animate-fade-in">
      {/* Results header */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Search Results</h2>
          <p className="text-gray-600 mt-1">
            Found <span className="font-semibold text-primary-600">{total}</span> matching clips
            {query && <span className="text-gray-400"> for "{query}"</span>}
          </p>
        </div>
        
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <button className="btn-secondary">
            View By Clips
          </button>
        </div>
      </div>

      {/* Results grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {clips.map((clip, index) => (
          <VideoClipCard
            key={`${clip.video_id}-${clip.timestamp_start}-${index}`}
            clip={clip}
            onClick={onClipClick}
          />
        ))}
      </div>
    </div>
  );
};

export default ResultsGrid;
