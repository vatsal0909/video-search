import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const searchClips = async (query, topK = 10) => {
  try {
    const response = await api.post('/hybrid-search', {
      query,
      top_k: topK,
    });
    return response.data;
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
    const response = await api.get('/videos/list');
    return response.data;
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
