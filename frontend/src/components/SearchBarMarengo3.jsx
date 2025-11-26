import React, { useState, useEffect, useRef } from 'react';
import { Search, X, Loader2, ChevronDown, ImagePlus as ImageIcon, ArrowRight } from 'lucide-react';

const SearchBarMarengo3 = ({ onSearch, isLoading, onSearchTypeChange, queryValue = '', onQueryChange }) => {
  const [query, setQuery] = useState(queryValue);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [imageError, setImageError] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    setQuery(queryValue || '');
  }, [queryValue]);

  const updateQuery = (value) => {
    setQuery(value);
    onQueryChange?.(value);
  };

  const [showDropdown, setShowDropdown] = useState(false);
  const [visual, setVisual] = useState(true);
  const [audio, setAudio] = useState(true);
  const [transcription, setTranscription] = useState(false);
  const [topK, setTopK] = useState(10);

  // Determine search type based on selections
  const getSearchType = () => {
    // Single modality selections
    if (visual && !audio && !transcription) return 'visual';
    if (!visual && audio && !transcription) return 'audio';
    if (!visual && !audio && transcription) return 'transcription';
    
    // UPDATED: 7 search options - specific combinations instead of intent-based
    // Two-modality combinations
    if (visual && audio && !transcription) return 'vector';
    if (visual && !audio && transcription) return 'vector';
    if (!visual && audio && transcription) return 'vector';
    
    // All three modalities → 'vector' (balanced search)
    if (visual && audio && transcription) return 'vector';
    
    // COMMENTED OUT: Old intent-based logic
    // // Any combination (including all three) → 'vector'
    // // Backend will use intent classification to determine modality focus
    // return 'vector'; // default for any multi-modality combination
    
    // Default fallback (if none selected, use vector)
    return 'vector';
  };

  const handle_submit = async (e) => {
    e.preventDefault();
    
    const searchType = getSearchType();
    
    // Support combined text + image search
    if (query.trim() || selectedImage) {
      console.log("Searching (Marengo 3):", {
        hasText: !!query.trim(),
        hasImage: !!selectedImage,
        searchType,
        topK
      });
      // Pass both query and image file to parent
      onSearch(query.trim() || null, searchType, topK, null, selectedImage);
    }
  };

  const clear_query = () => {
    updateQuery('');
  };

  const handleImageSelect = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Clear previous errors
    setImageError(null);

    // Validate file type
    const validImageTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    if (!validImageTypes.includes(file.type)) {
      setImageError(`Invalid image type: ${file.type}. Supported: JPEG, PNG, GIF, WebP`);
      return;
    }

    // Validate file size (5MB limit)
    const maxSize = 5 * 1024 * 1024;
    if (file.size > maxSize) {
      setImageError(`Image exceeds 5MB limit. Size: ${(file.size / 1024 / 1024).toFixed(2)}MB`);
      return;
    }

    // Set selected image - DO NOT clear text query
    setSelectedImage(file);

    // Create preview
    const reader = new FileReader();
    reader.onload = (event) => {
      setImagePreview(event.target?.result);
    };
    reader.readAsDataURL(file);
  };

  const removeImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setImageError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleVisualChange = (e) => {
    const newVisual = e.target.checked;
    setVisual(newVisual);
    onSearchTypeChange?.(getSearchType());
  };

  const handleAudioChange = (e) => {
    const newAudio = e.target.checked;
    setAudio(newAudio);
    onSearchTypeChange?.(getSearchType());
  };

  const handleTranscriptionChange = (e) => {
    const newTranscription = e.target.checked;
    setTranscription(newTranscription);
    onSearchTypeChange?.(getSearchType());
  };

  return (
    <form onSubmit={handle_submit} className="w-full">
      {/* Main Row - Search bar takes more space */}
      <div className="flex gap-3 items-center">
        {/* Main Search Container - Larger and more prominent */}
        <div className="relative flex-1">
          {/* Search Input */}
          <div className={`relative transition-all duration-300 ${
            selectedImage ? 'min-h-32 p-4' : 'h-16'
          } rounded-3xl border border-gray-200 bg-white shadow-sm hover:shadow-md flex flex-col`}>
            
            {/* Image Preview - Show alongside text input */}
            {imagePreview && (
              <div className="flex gap-3 mb-2 items-center">
                {/* Image Thumbnail */}
                <div className="relative flex-shrink-0 rounded-lg overflow-hidden bg-gray-100 w-24 h-24 flex items-center justify-center">
                  <img 
                    src={imagePreview} 
                    alt="Selected" 
                    className="max-w-full max-h-full object-contain"
                  />
                  {/* Remove button - top right */}
                  <button
                    type="button"
                    onClick={removeImage}
                    className="absolute top-1 right-1 bg-red-500 hover:bg-red-600 text-white rounded-full p-1 transition-colors shadow-md"
                    title="Remove image"
                  >
                    <X size={14} />
                  </button>
                </div>
                
              </div>
            )}
            
            {/* Text Input - Always visible */}
            <div className="relative w-full h-full flex items-center px-2">
              <div className="absolute left-11 text-gray-400 flex-shrink-0">
                <Search size={22} />
              </div>
              <input
                type="text"
                value={query}
                onChange={(e) => updateQuery(e.target.value)}
                placeholder={selectedImage ? "Add text to refine search..." : "Search videos, actions, or objects..."}
                className="w-full h-full pl-16 pr-20 text-lg bg-transparent focus:outline-none"
                disabled={isLoading}
              />
              
              {/* Image Upload Button */}
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className={`absolute left-2 flex items-center justify-center w-8 h-8 rounded-lg transition-colors ${
                  selectedImage 
                    ? 'bg-green-200 hover:bg-green-300 text-green-600' 
                    : 'bg-gray-200 hover:bg-gray-300 text-gray-600'
                }`}
                disabled={isLoading}
                title={selectedImage ? "Change image" : "Upload image to search"}
              >
                <ImageIcon size={16} />
              </button>
              
              {query && !isLoading && (
                <button
                  type="button"
                  onClick={clear_query}
                  className="absolute right-6 text-gray-400 hover:text-gray-600 transition-colors"
                >
                  <X size={20} />
                </button>
              )}
              
              {isLoading && (
                <div className="absolute right-6">
                  <Loader2 size={20} className="animate-spin text-gray-600" />
                </div>
              )}
            </div>
          </div>

          {/* Error message - Outside search bar */}
          {imageError && (
            <div className="mt-2 p-3 bg-red-50 border border-red-200 rounded-xl">
              <p className="text-red-600 text-sm">{imageError}</p>
            </div>
          )}

          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/gif,image/webp"
            onChange={handleImageSelect}
            className="hidden"
          />
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isLoading || (!query.trim() && !selectedImage)}
          className="flex items-center justify-center w-14 h-14 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-300 text-white rounded-2xl transition-colors shadow-sm hover:shadow-md disabled:cursor-not-allowed flex-shrink-0"
          title="Search"
        >
          {isLoading ? (
            <Loader2 size={20} className="animate-spin" />
          ) : (
            <ArrowRight size={20} />
          )}
        </button>

        {/* Options Button - Relative for dropdown positioning */}
        <div className="relative flex-shrink-0">
          <button
            type="button"
            onClick={() => setShowDropdown(!showDropdown)}
            className="flex items-center gap-2 px-3 py-4 bg-gray-100 hover:bg-gray-200 text-gray-600 rounded-2xl border border-gray-200 transition-colors"
            disabled={isLoading}
            title="Search options"
          >
            <div className="w-5 h-5 flex items-center justify-center flex-shrink-0">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
              </svg>
            </div>
            <ChevronDown size={18} />
          </button>

          {/* Dropdown Menu - Positioned below options button */}
          {showDropdown && (
            <div className="absolute right-0 top-full mt-2 w-56 bg-white rounded-lg border border-gray-200 shadow-lg p-3 z-50">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">Search Options</h3>
              
              <div className="space-y-2">
                {/* Visual Checkbox */}
                <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-50 p-1.5 rounded">
                  <input
                    type="checkbox"
                    checked={visual}
                    onChange={handleVisualChange}
                    className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500"
                  />
                  <span className="text-sm font-medium text-gray-700">Visual</span>
                </label>

                {/* Audio Checkbox */}
                <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-50 p-1.5 rounded">
                  <input
                    type="checkbox"
                    checked={audio}
                    onChange={handleAudioChange}
                    className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500"
                  />
                  <span className="text-sm font-medium text-gray-700">Audio</span>
                </label>

                {/* Transcription Checkbox */}
                <label className="flex items-center gap-2 cursor-pointer hover:bg-gray-50 p-1.5 rounded">
                  <input
                    type="checkbox"
                    checked={transcription}
                    onChange={handleTranscriptionChange}
                    className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500"
                  />
                  <span className="text-sm font-medium text-gray-700">Transcription</span>
                </label>
              </div>

              {/* Search Type Display */}
              <div className="mt-2 pt-2 border-t border-gray-200">
                <p className="text-xs text-gray-500">
                  Type: <span className="font-semibold text-gray-700">{getSearchType()}</span>
                </p>
              </div>

              {/* Combined Search Info */}
              <div className="mt-2 pt-2 border-t border-gray-200">
                <p className="text-xs text-gray-600">
                  💡 Tip: Use text + image together for multimodal search
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </form>
  );
};

export default SearchBarMarengo3;
