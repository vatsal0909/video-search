import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_GATEWAY_URL = import.meta.env.VITE_API_GATEWAY_URL
const BACKEND_ALB_URL = import.meta.env.VITE_BACKEND_ALB_URL

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Old searchClips - commented out
// export const searchClips = async (query, topK = 10) => {
//   try {
//     const response = await api.post('/hybrid-search', {
//       query,
//       top_k: topK,
//     });
//     return response.data;
//   } catch (error) {
//     console.error('Error searching clips:', error);
//     throw error;
//   }
// };

// New searchClips using API Gateway endpoint
export const searchClips = async (query, topK = 10) => {
  try {
    const response = await fetch(`${BACKEND_ALB_URL}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query_text: query,
        top_k: topK,
        search_type: "vector"
      }),
    });
    
    // Check if response is ok
    if (!response.ok) {
      throw new Error(`API returned status code: ${response.status}`);
    }
    
    // Parse the response body
    const data = await response.json();
    console.log('API Response:', data);
    
    // Parse Lambda response format: { statusCode, body: "...json.dumps..." }
    if (data) {
      // Parse the body string which contains JSON from python json.dumps
      return data
    } else {
      throw new Error(`API returned error`);
    }
  } catch (error) {
    console.error('Error searching clips:', error);
    throw error;
  }
};

export const processVideo = async (videoUrl) => {
  try {
    const response = await api.post('/process-video', {
      video_url: videoUrl,
    });
    return response.data;
  } catch (error) {
    console.error('Error processing video:', error);
    throw error;
  }
};

export const getVideoStatus = async (videoId) => {
  try {
    const response = await api.get(`/video-status/${videoId}`);
    return response.data;
  } catch (error) {
    console.error('Error getting video status:', error);
    throw error;
  }
};

export const askQuestion = async (question) => {
  try {
    const response = await api.post('/ask', {
      question,
    });
    return response.data;
  } catch (error) {
    console.error('Error asking question:', error);
    throw error;
  }
};

export const listAllVideos = async () => {
  try {
    const response = await fetch(`${BACKEND_ALB_URL}/list`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    let data = await response.json();
    console.log(data)
    return data;
  } catch (error) {
    console.error('Error listing videos:', error);
    throw error;
  }
};

export const getVideoDetails = async (videoId) => {
  try {
    const response = await api.get(`/videos/${videoId}`);
    return response.data;
  } catch (error) {
    console.error('Error getting video details:', error);
    throw error;
  }
};

export default api;
