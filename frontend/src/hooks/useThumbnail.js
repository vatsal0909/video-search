import { useState, useEffect } from 'react';
import { get_or_generate_thumbnail } from '../utils/thumbnailGenerator';

/**
 * Custom hook to load video thumbnails
 * Prioritizes thumbnail_path (presigned URL from backend), falls back to generating from video
 */
export const use_thumbnail = (videoUrl, videoId, timestamp, thumbnailPath = null) => {
  const [thumbnail, setThumbnail] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;

    const load_thumbnail = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // Priority 1: Use thumbnail_path if available (presigned URL from backend)
        if (thumbnailPath) {
          // console.log('Using thumbnail_path from backend:', thumbnailPath);
          if (isMounted) {
            setThumbnail(thumbnailPath);
            setIsLoading(false);
          }
          return;
        }

        // Priority 2: Generate thumbnail from video if available
        if (!videoUrl || videoId === undefined || timestamp === undefined) {
          setIsLoading(false);
          return;
        }

        console.log('Generating thumbnail from video:', videoUrl);
        const thumbnailUrl = await get_or_generate_thumbnail(videoUrl, videoId, timestamp);
        
        if (isMounted) {
          setThumbnail(thumbnailUrl);
          setIsLoading(false);
        }
      } catch (err) {
        if (isMounted) {
          setError(err);
          setIsLoading(false);
          console.error('Thumbnail loading error:', err.message);
        }
      }
    };

    load_thumbnail();

    return () => {
      isMounted = false;
    };
  }, [videoUrl, videoId, timestamp, thumbnailPath]);

  return { thumbnail, isLoading, error };
};
