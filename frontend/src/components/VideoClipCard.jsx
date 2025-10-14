import React from 'react';
import { Play, Clock, TrendingUp, Loader2 } from 'lucide-react';
import { formatTimestamp } from '../utils/formatTime';
import { use_thumbnail } from '../hooks/useThumbnail';

const VideoClipCard = ({ clip, onClick }) => {
  const { video_id, video_path, timestamp_start, timestamp_end, clip_text, score, presigned_url } = clip;

  // Normalize score to 0-100 range
  const rawScore = typeof score === 'number' ? score : 0;
  const normalizedScore = rawScore > 1 ? rawScore : rawScore * 100;
  const clampedScore = Math.max(0, Math.min(normalizedScore, 100));

  const evaluationScore = Math.min(clampedScore, 80);

  let confidenceLabel = 'LOW';
  let indicatorBg = 'bg-red-500';

  if (evaluationScore >= 40 && evaluationScore < 60) {
    confidenceLabel = 'MEDIUM';
    indicatorBg = 'bg-yellow-500';
  } else if (evaluationScore >= 60) {
    confidenceLabel = 'HIGH';
    indicatorBg = 'bg-green-500';
  }

  // Load thumbnail from video - use presigned_url if available, fallback to video_path
  const videoUrl = presigned_url || video_path;
  const { thumbnail, isLoading: thumbnailLoading, error: thumbnailError } = use_thumbnail(
    videoUrl, 
    video_id, 
    timestamp_start
  );

  return (
    <div 
      className="bg-white rounded-2xl shadow-md hover:shadow-xl hover:border-blue-200 border border-transparent transition-all duration-300 overflow-hidden cursor-pointer group"
      onClick={() => onClick(clip)}
    >
      {/* Video Thumbnail */}
      <div className="relative h-64 bg-gray-200 flex items-center justify-center overflow-hidden">
        {/* Loading state */}
        {thumbnailLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-200">
            <Loader2 size={40} className="text-gray-400 animate-spin" />
          </div>
        )}
        
        {/* Error state */}
        {thumbnailError && !thumbnail && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-200 text-gray-600 p-4">
            <div className="text-red-400 text-sm text-center mb-2">
              Unable to load thumbnail
            </div>
            <div className="text-xs text-gray-400 text-center">
              CORS not enabled on video
            </div>
          </div>
        )}
        
        {/* Actual thumbnail */}
        {thumbnail && (
          <img 
            src={thumbnail} 
            alt={`Thumbnail at ${formatTimestamp(timestamp_start)}`}
            className="absolute inset-0 w-full h-full object-cover"
            loading="lazy"
          />
        )}

        {/* Confidence indicator */}
        <div className={`absolute top-2 right-2 px-3 py-1 rounded-full text-xs font-semibold text-white ${indicatorBg}`}>
          {confidenceLabel}
        </div>
        
        {/* Timestamp overlay */}
        <div className="absolute bottom-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
          {formatTimestamp(timestamp_start)}
        </div>
      </div>
      
      {/* Card content */}
      {/* <div className="p-4">
        <p className="text-md font-semibold truncate text-gray-900">
          {clip_text || 'Video clip segment'}
        </p>
      </div> */}
    </div>
  );
};

export default VideoClipCard;
