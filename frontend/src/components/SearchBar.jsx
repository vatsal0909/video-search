import React, { useState } from 'react';
import { Search, X, Loader2 } from 'lucide-react';

const SearchBar = ({ onSearch, isLoading }) => {
  const [query, setQuery] = useState('');

  const handle_submit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query);
    }
  };

  const clear_query = () => {
    setQuery('');
  };

  return (
    <form onSubmit={handle_submit} className="w-full">
      <div className="relative">
        <div className="absolute left-6 top-1/2 -translate-y-1/2 text-gray-400">
          <Search size={20} />
        </div>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search videos, actions, or objects..."
          className="w-full py-4 pl-14 pr-16 text-lg rounded-2xl border border-blue-200 focus:outline-none focus:ring-4 focus:ring-blue-200 focus:border-blue-400 shadow-sm hover:shadow-md transition-all"
          disabled={isLoading}
        />
        
        {query && !isLoading && (
          <button
            type="button"
            onClick={clear_query}
            className="absolute right-6 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X size={20} />
          </button>
        )}
        
        {isLoading && (
          <div className="absolute right-6 top-1/2 -translate-y-1/2">
            <Loader2 size={20} className="animate-spin text-gray-600" />
          </div>
        )}
      </div>
    </form>
  );
};

export default SearchBar;
