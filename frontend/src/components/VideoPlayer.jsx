import { useRef, useEffect } from 'react';
import { X, ExternalLink } from 'lucide-react';
import { formatTimestamp } from '../utils/formatTime';
import JsonDisplay from './JsonDisplay';

const VideoPlayer = ({ clip, onClose }) => {
  const videoRef = useRef(null);

  useEffect(() => {
    if (videoRef.current && clip) {
      videoRef.current.currentTime = clip.timestamp_start;
      videoRef.current.play();
    }
  }, [clip]);

  if (!clip) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fade-in">
      <div className="bg-white rounded-2xl shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-hidden animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Video Clip</h3>
            <p className="text-sm text-gray-500">
              {formatTimestamp(clip.timestamp_start)} - {formatTimestamp(clip.timestamp_end)}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X size={24} />
          </button>
        </div>

        <div className="flex flex-col lg:flex-row">
          <div className="lg:w-3/5">
            {/* Video Player */}
            <div className="bg-black">
              <video
                ref={videoRef}
                src={`${clip.video_path}#t=${clip.timestamp_start},${clip.timestamp_end}`}
                controls
                className="w-full aspect-video"
                controlsList="nodownload"
              >
                Your browser does not support the video tag.
              </video>
            </div>

            {/* Clip Info */}
            <div className="p-6 space-y-4">
              <div>
                <h4 className="text-sm font-semibold text-gray-700 mb-2">Clip Information</h4>
                <p className="text-gray-600">{clip.clip_text}</p>
              </div>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Start Time:</span>
                  <span className="ml-2 font-medium">{formatTimestamp(clip.timestamp_start)}</span>
                </div>
                <div>
                  <span className="text-gray-500">End Time:</span>
                  <span className="ml-2 font-medium">{formatTimestamp(clip.timestamp_end)}</span>
                </div>
                <div>
                  <span className="text-gray-500">Relevance Score:</span>
                  <span className="ml-2 font-medium">{(clip.score * 100).toFixed(1)}%</span>
                </div>
                <div>
                  <span className="text-gray-500">Video ID:</span>
                  <span className="ml-2 font-mono text-xs">{clip.video_id}</span>
                </div>
              </div>

              <div className="flex gap-3 pt-4">
                <a
                  href={clip.video_path}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-secondary flex items-center gap-2"
                >
                  <ExternalLink size={16} />
                  Open Full Video
                </a>
              </div>
            </div>
          </div>

          {/* JSON Display */}
          <div className="lg:w-2/5 p-6 border-l border-gray-200 overflow-auto">
            <h4 className="text-sm font-semibold text-gray-700 mb-3">Clip Data</h4>
            <JsonDisplay data={clip} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoPlayer;
