import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const BACKEND_ALB_URL = import.meta.env.VITE_BACKEND_ALB_URL

const BACKEND_URL = process.env.NODE_ENV === 'development' ? API_BASE_URL : BACKEND_ALB_URL

const api = axios.create({
  baseURL: BACKEND_URL,
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

// Helper function to convert file to base64
const fileToBase64 = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64String = reader.result.split(',')[1]; // Extract base64 part only
      resolve(base64String);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

// Unified search function - handles both text and image searches
export const searchClips = async (query, topK = 10, searchType = 'vector', imageFile = null) => {
  try {
    // Validate that at least one input is provided
    if (!query && !imageFile) {
      throw new Error('Either query text or image file is required');
    }

    let requestBody = {
      top_k: topK,
      search_type: searchType
    };

    // Handle text search
    if (query) {
      console.log('🔍 Starting text search:', {
        query,
        topK,
        searchType
      });
      requestBody.query_text = query;
    }

    // Handle image search
    if (imageFile) {
      console.log('📷 Starting image search with file:', {
        name: imageFile.name,
        type: imageFile.type,
        size: `${(imageFile.size / 1024).toFixed(2)}KB`,
        topK
      });

      // Validate file type
      const validImageTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
      if (!validImageTypes.includes(imageFile.type)) {
        throw new Error(`Invalid image type: ${imageFile.type}. Supported types: JPEG, PNG, GIF, WebP`);
      }

      // Validate file size (max 5MB)
      const maxSize = 5 * 1024 * 1024;
      if (imageFile.size > maxSize) {
        throw new Error(`Image file exceeds 5MB limit. Size: ${(imageFile.size / 1024 / 1024).toFixed(2)}MB`);
      }

      console.log('✓ File validation passed');
      console.log('🔄 Converting image to base64...');
      
      const base64String = await fileToBase64(imageFile);
      console.log(`✓ Base64 conversion complete. Length: ${base64String.length} characters`);
      
      requestBody.image_base64 = base64String;
    }

    const response = await fetch(`${BACKEND_URL}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });
    
    // Check if response is ok
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(`API returned status code: ${response.status}. ${errorData.detail || ''}`);
    }
    
    // Parse the response body
    const data = await response.json();
    console.log('✓ Search completed successfully');
    console.log(`  - Results found: ${data.clips?.length || 0}`);
    console.log(`  - Total: ${data.total}`);
    
    return data;
  } catch (error) {
    console.error('❌ Error searching clips:', error);
    throw error;
  }
};

// Legacy wrapper for image search - calls unified searchClips function
export const searchClipsWithImage = async (imageFile, topK = 10, searchType = 'vector') => {
  return searchClips(null, topK, searchType, imageFile);
};

// ============ MARENGO 3 SEARCH FUNCTIONS ============

// Unified search function for Marengo 3 - handles text, image, or combined text+image searches
// Supported search types: 'vector', 'visual', 'audio', 'transcription'
export const searchClipsMarengo3 = async (query = null, topK = 10, searchType = 'vector', imageFile = null) => {
  try {
    // Validate that at least one input is provided
    if (!query && !imageFile) {
      throw new Error('Either query text or image file is required');
    }

    // Validate search type
    const validSearchTypes = ['vector', 'visual', 'audio', 'transcription', 'vector_visual_audio', 'vector_audio_transcription', 'vector_visual_transcription'];
    if (!validSearchTypes.includes(searchType)) {
      throw new Error(`Invalid search type: ${searchType}. Supported types: ${validSearchTypes.join(', ')}`);
    }

    let requestBody = {
      top_k: topK,
      search_type: searchType
    };

    // Determine search type for logging
    let searchInputType = '';
    if (query && imageFile) {
      searchInputType = 'multimodal (text + image)';
      console.log('🔄 Starting multimodal search (Marengo 3):', {
        query,
        imageFile: imageFile.name,
        topK,
        searchType
      });
    } else if (imageFile) {
      searchInputType = 'image-only';
      console.log('📷 Starting image-only search (Marengo 3) with file:', {
        name: imageFile.name,
        type: imageFile.type,
        size: `${(imageFile.size / 1024).toFixed(2)}KB`,
        topK,
        searchType
      });
    } else {
      searchInputType = 'text-only';
      console.log('🔍 Starting text-only search (Marengo 3):', {
        query,
        topK,
        searchType
      });
    }

    // Handle text search
    if (query) {
      requestBody.query_text = query;
    }

    // Handle image search
    if (imageFile) {
      // Validate file type
      const validImageTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
      if (!validImageTypes.includes(imageFile.type)) {
        throw new Error(`Invalid image type: ${imageFile.type}. Supported types: JPEG, PNG, GIF, WebP`);
      }

      // Validate file size (max 5MB)
      const maxSize = 5 * 1024 * 1024;
      if (imageFile.size > maxSize) {
        throw new Error(`Image file exceeds 5MB limit. Size: ${(imageFile.size / 1024 / 1024).toFixed(2)}MB`);
      }

      console.log('✓ File validation passed');
      console.log('🔄 Converting image to base64...');
      
      const base64String = await fileToBase64(imageFile);
      console.log(`✓ Base64 conversion complete. Length: ${base64String.length} characters`);
      
      requestBody.image_base64 = base64String;
    }

    console.log(`📤 Sending ${searchInputType} request to /search-3 (searchType: ${searchType})`);
    const response = await fetch(`${BACKEND_URL}/search-3`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });
    
    // Check if response is ok
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(`API returned status code: ${response.status}. ${errorData.detail || ''}`);
    }
    
    // Parse the response body
    const data = await response.json();
    console.log('✓ Search (Marengo 3) completed successfully');
    console.log(`  - Search input type: ${searchInputType}`);
    console.log(`  - Search type: ${searchType}`);
    console.log(`  - Results found: ${data.clips?.length || 0}`);
    console.log(`  - Total: ${data.total}`);
    console.log(`  - Search Intent: ${data.classified_intent}`);
    
    return data;
  } catch (error) {
    console.error('❌ Error searching clips (Marengo 3):', error);
    throw error;
  }
};

// Wrapper for Marengo 3 image-only search
export const searchClipsWithImageMarengo3 = async (imageFile, topK = 10, searchType = 'vector') => {
  return searchClipsMarengo3(null, topK, searchType, imageFile);
};

// Wrapper for Marengo 3 text-only search
export const searchClipsTextMarengo3 = async (query, topK = 10, searchType = 'vector') => {
  return searchClipsMarengo3(query, topK, searchType, null);
};

// Individual search type wrappers for Marengo 3
export const searchClipsVisualMarengo3 = async (query, topK = 10, imageFile = null) => {
  return searchClipsMarengo3(query, topK, 'visual', imageFile);
};

export const searchClipsAudioMarengo3 = async (query, topK = 10, imageFile = null) => {
  return searchClipsMarengo3(query, topK, 'audio', imageFile);
};

export const searchClipsTranscriptionMarengo3 = async (query, topK = 10, imageFile = null) => {
  return searchClipsMarengo3(query, topK, 'transcription', imageFile);
};

// # COMMENTED OUT: Vector search type combinations replaced by intent classification
// # Frontend now only sends: 'vector', 'visual', 'audio', 'transcription'
// # Backend uses intent classification when 'vector' is received
// export const searchClipsVectorVisualAudioMarengo3 = async (query, topK = 10, imageFile = null) => {
//   return searchClipsMarengo3(query, topK, 'vector_visual_audio', imageFile);
// };
//
// export const searchClipsVectorVisualTranscriptionMarengo3 = async (query, topK = 10, imageFile = null) => {
//   return searchClipsMarengo3(query, topK, 'vector_visual_transcription', imageFile);
// };
//
// export const searchClipsVectorAudioTranscriptionMarengo3 = async (query, topK = 10, imageFile = null) => {
//   return searchClipsMarengo3(query, topK, 'vector_audio_transcription', imageFile);
// };


export const listAllVideos = async () => {
  try {
    const response = await fetch(`${BACKEND_URL}/list`, {
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

export const getPresignedUploadUrl = async (filename) => {
  try {
    const response = await fetch(`${BACKEND_URL}/generate-upload-presigned-url?filename=${encodeURIComponent(filename)}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      throw new Error(`Failed to generate presigned URL: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Presigned URL generated:', {
      s3_key: data.s3_key,
      expires_in: data.expires_in
    });
    
    return data;
  } catch (error) {
    console.error('Error getting presigned upload URL:', error);
    throw error;
  }
};

export default api;
