// Queue to limit concurrent thumbnail generation
// class ThumbnailQueue {
//   constructor(maxConcurrent = 2) {
//     this.maxConcurrent = maxConcurrent;
//     this.running = 0;
//     this.queue = [];
//   }

//   async add(fn) {
//     while (this.running >= this.maxConcurrent) {
//       await new Promise(resolve => {
//         this.queue.push(resolve);
//       });
//     }

//     this.running++;
//     try {
//       return await fn();
//     } finally {
//       this.running--;
//       const resolve = this.queue.shift();
//       if (resolve) resolve();
//     }
//   }
// }

// No concurrency
class ThumbnailQueue {
  constructor() {
    this.running = 0;
    this.queue = [];
  }

  async add(fn) {
    this.running++;
    try {
      return await fn();
    } finally {
      this.running--;
      const resolve = this.queue.shift();
      if (resolve) resolve();
    }
  }
}

const thumbnailQueue = new ThumbnailQueue(); // Max 2 concurrent generations

/**
 * Generate a thumbnail from a video at a specific timestamp
 * @param {string} videoUrl - URL of the video
 * @param {number} timestamp - Timestamp in seconds
 * @returns {Promise<string>} - Data URL of the thumbnail
 */
export const generate_thumbnail = (videoUrl, timestamp) => {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.crossOrigin = 'anonymous';
    video.preload = 'metadata';
    video.muted = true;
    
    let hasResolved = false;
    
    video.addEventListener('loadedmetadata', () => {
      if (hasResolved) return;
      // Ensure timestamp is within video duration
      const seekTime = Math.min(timestamp, video.duration - 0.1);
      video.currentTime = seekTime;
    });
    
    video.addEventListener('seeked', () => {
      if (hasResolved) return;
      hasResolved = true;
      
      try {
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 360;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        const thumbnailUrl = canvas.toDataURL('image/jpeg', 0.85);
        resolve(thumbnailUrl);
      } catch (error) {
        reject(new Error(`Failed to capture frame: ${error.message}`));
      } finally {
        video.remove();
      }
    });
    
    video.addEventListener('error', (e) => {
      if (hasResolved) return;
      hasResolved = true;
      
      const errorMsg = video.error ? 
        `Video error (code ${video.error.code}): ${video.error.message}` : 
        'Failed to load video';
      
      reject(new Error(errorMsg));
      video.remove();
    });
    
    // Increased timeout for slower connections
    setTimeout(() => {
      if (!hasResolved) {
        hasResolved = true;
        reject(new Error('Thumbnail generation timeout'));
        video.remove();
      }
    }, 30000); // Increased to 30 seconds
    
    video.src = videoUrl;
  });
};

/**
 * Create a thumbnail cache key
 */
export const get_thumbnail_cache_key = (videoId, timestamp) => {
  return `thumbnail_${videoId}_${Math.floor(timestamp)}`;
};

/**
 * Get thumbnail from cache or generate new one (with queue management)
 */
export const get_or_generate_thumbnail = async (videoUrl, videoId, timestamp) => {
  const cacheKey = get_thumbnail_cache_key(videoId, timestamp);
  
  // Check sessionStorage cache
  try {
    const cached = sessionStorage.getItem(cacheKey);
    if (cached) {
      return cached;
    }
  } catch (e) {
    console.warn('SessionStorage not available:', e);
  }
  
  // Use queue to limit concurrent generations
  try {
    const thumbnail = await thumbnailQueue.add(() => 
      generate_thumbnail(videoUrl, timestamp)
    );
    
    // Cache the thumbnail
    try {
      sessionStorage.setItem(cacheKey, thumbnail);
    } catch (e) {
      console.warn('Failed to cache thumbnail:', e);
    }
    
    return thumbnail;
  } catch (error) {
    console.error('Failed to generate thumbnail:', error.message);
    throw error;
  }
};
