/**
 * Runtime Configuration Manager
 * 
 * This module handles dynamic loading of configuration from config.json
 * at runtime, allowing backend URL updates without frontend rebuilds.
 * 
 * Features:
 * - Singleton pattern to ensure config is loaded once
 * - Development vs production environment detection
 * - Graceful fallback to localhost in development
 * - Config validation and error handling
 * - Promise-based loading with caching
 */

class RuntimeConfig {
  constructor() {
    this.config = null;
    this.loadPromise = null;
    this.isLoading = false;
  }

  /**
   * Load configuration from /config.json
   * Uses singleton pattern - only loads once per session
   * @returns {Promise<Object>} Configuration object
   */
  async load() {
    // If already loaded, return cached config
    if (this.config) {
      console.log('✓ Using cached configuration');
      return this.config;
    }

    // If currently loading, return existing promise
    if (this.loadPromise) {
      console.log('⏳ Configuration loading in progress, waiting...');
      return this.loadPromise;
    }

    // Start loading process
    this.isLoading = true;
    this.loadPromise = this._loadConfig();

    try {
      this.config = await this.loadPromise;
      console.log('✓ Configuration loaded successfully');
      return this.config;
    } catch (error) {
      // Reset loading state on error to allow retry
      this.loadPromise = null;
      this.isLoading = false;
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  /**
   * Internal method to fetch and validate configuration
   * @private
   */
  async _loadConfig() {
    const isDevelopment = this.isProduction() === false;
    
    try {
      console.log('📡 Fetching configuration from /config.json...');
      
      const response = await fetch('/config.json', {
        method: 'GET',
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        }
      });

      if (!response.ok) {
        if (response.status === 404) {
          if (isDevelopment) {
            console.log('⚠️ config.json not found, using development fallback');
            return this._getDevelopmentConfig();
          } else {
            throw new Error('Configuration file not found. Please ensure config.json is deployed to the root of your frontend.');
          }
        }
        throw new Error(`Failed to fetch configuration: ${response.status} ${response.statusText}`);
      }

      const config = await response.json();
      console.log('✓ Configuration fetched successfully');
      
      // Validate configuration structure
      this._validateConfig(config);
      
      console.log('✓ Configuration validation passed');
      console.log(`  - Backend URL: ${config.backendUrl}`);
      console.log(`  - Environment: ${config.environment || 'not specified'}`);
      console.log(`  - API Version: ${config.apiVersion || 'not specified'}`);
      
      return config;

    } catch (error) {
      if (isDevelopment) {
        console.log('⚠️ Failed to load config.json in development, using fallback');
        console.log(`  - Error: ${error.message}`);
        return this._getDevelopmentConfig();
      } else {
        console.error('❌ Failed to load configuration in production:', error.message);
        throw new Error(`Configuration loading failed: ${error.message}`);
      }
    }
  }

  /**
   * Get development fallback configuration
   * @private
   */
  _getDevelopmentConfig() {
    console.log('🔧 Using development configuration');
    console.log('⚠️ WARNING: Using localhost backend. For deployed backend, ensure config.json is generated with CloudFormation stack output.');
    return {
      backendUrl: 'http://localhost:8000',
      apiVersion: 'v1',
      environment: 'development',
      features: {
        marengo3Enabled: true,
        uploadEnabled: true,
        maxUploadSizeMB: 5
      },
      endpoints: {
        search: '/search',
        search3: '/search-3',
        list: '/list',
        upload: '/generate-upload-presigned-url',
        health: '/health'
      },
      metadata: {
        lastUpdated: new Date().toISOString(),
        version: '1.0.0-dev',
        source: 'development-fallback'
      }
    };
  }

  /**
   * Validate configuration structure
   * @private
   */
  _validateConfig(config) {
    const requiredFields = ['backendUrl'];
    const missingFields = requiredFields.filter(field => !config[field]);
    
    if (missingFields.length > 0) {
      throw new Error(`Configuration validation failed. Missing required fields: ${missingFields.join(', ')}`);
    }

    // Validate backendUrl format
    try {
      new URL(config.backendUrl);
    } catch (error) {
      throw new Error(`Invalid backendUrl format: ${config.backendUrl}`);
    }

    // Warn about missing optional fields
    const recommendedFields = ['apiVersion', 'environment', 'endpoints'];
    const missingRecommended = recommendedFields.filter(field => !config[field]);
    if (missingRecommended.length > 0) {
      console.warn('⚠️ Configuration missing recommended fields:', missingRecommended.join(', '));
    }
  }

  /**
   * Get the backend URL for API calls
   * Throws error if configuration not loaded
   * @returns {string} Backend URL
   */
  getBackendUrl() {
    if (!this.config) {
      throw new Error('Configuration not loaded. Call runtimeConfig.load() first.');
    }
    return this.config.backendUrl;
  }

  /**
   * Get the full configuration object
   * @returns {Object|null} Configuration object or null if not loaded
   */
  getConfig() {
    return this.config;
  }

  /**
   * Check if running in production environment
   * @returns {boolean} True if production, false if development
   */
  isProduction() {
    // Check multiple indicators for production environment
    const hostname = window.location.hostname;
    const protocol = window.location.protocol;
    
    // Development indicators
    const isDevelopment = (
      hostname === 'localhost' ||
      hostname === '127.0.0.1' ||
      hostname.startsWith('192.168.') ||
      hostname.startsWith('10.') ||
      hostname.includes('dev') ||
      protocol === 'file:' ||
      (window.location.port && ['3000', '5173', '8080'].includes(window.location.port))
    );

    return !isDevelopment;
  }

  /**
   * Get endpoint URL for a specific API endpoint
   * @param {string} endpointName - Name of the endpoint (e.g., 'search', 'upload')
   * @returns {string} Full endpoint URL
   */
  getEndpointUrl(endpointName) {
    if (!this.config) {
      throw new Error('Configuration not loaded. Call runtimeConfig.load() first.');
    }

    const baseUrl = this.config.backendUrl;
    const endpoints = this.config.endpoints || {};
    const endpoint = endpoints[endpointName];

    if (!endpoint) {
      console.warn(`⚠️ Endpoint '${endpointName}' not found in configuration, using default`);
      return `${baseUrl}/${endpointName}`;
    }

    return `${baseUrl}${endpoint}`;
  }

  /**
   * Check if a feature is enabled
   * @param {string} featureName - Name of the feature
   * @returns {boolean} True if enabled, false otherwise
   */
  isFeatureEnabled(featureName) {
    if (!this.config || !this.config.features) {
      return true; // Default to enabled if no config
    }
    return this.config.features[featureName] !== false;
  }

  /**
   * Reset configuration (for testing or manual reload)
   */
  reset() {
    console.log('🔄 Resetting configuration cache');
    this.config = null;
    this.loadPromise = null;
    this.isLoading = false;
  }
}

// Export singleton instance
export default new RuntimeConfig();