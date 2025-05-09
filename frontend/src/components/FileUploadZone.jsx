import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ResetWeightsButton from './ResetWeightsButton';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

const FileUploadZone = ({ onSessionCreated }) => {
  const [files, setFiles] = useState([]);
  const [filePreviewUrls, setFilePreviewUrls] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [fetchingPublic, setFetchingPublic] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [processingComplete, setProcessingComplete] = useState(false);
  const [publicFetchComplete, setPublicFetchComplete] = useState(false);
  const [detectedLocations, setDetectedLocations] = useState([]);
  const [memoryStats, setMemoryStats] = useState({ user: 0, public: 0 });
  const [error, setError] = useState(null);
  const [resetMessage, setResetMessage] = useState(null);
  const [currentStep, setCurrentStep] = useState(1); // 1: Upload, 2: Process, 3: Fetch public, 4: Complete
  
  // API client configuration
  const api = axios.create({
    baseURL: process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000',
    timeout: 15000,
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
  });

  // Create a session when the component loads
  useEffect(() => {
    const createSession = async () => {
      try {
        const response = await api.post('/api/session/create');
        if (response.status === 200) {
          setSessionId(response.data.session_id);
          console.log(`Session created: ${response.data.session_id}`);
        } else {
          setError('Failed to create a session');
        }
      } catch (error) {
        console.error('Session creation error:', error);
        setError(`Error: ${error.message}`);
      }
    };
    
    createSession();
  }, []);

  // Handle file selection via input
  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles([...files, ...selectedFiles]);
    
    // Create preview URLs
    const newPreviewUrls = selectedFiles.map(file => {
      return {
        file,
        url: URL.createObjectURL(file)
      };
    });
    
    setFilePreviewUrls([...filePreviewUrls, ...newPreviewUrls]);
  };

  // Remove a file from the list
  const removeFile = (fileToRemove, index) => {
    // Remove file from files array
    setFiles(files.filter((_, i) => i !== index));
    
    // Remove preview URL and revoke object URL to prevent memory leaks
    const previewToRemove = filePreviewUrls[index];
    if (previewToRemove && previewToRemove.url) {
      URL.revokeObjectURL(previewToRemove.url);
    }
    setFilePreviewUrls(filePreviewUrls.filter((_, i) => i !== index));
  };

  const uploadFiles = async () => {
    if (!sessionId || files.length === 0) {
      console.log('Cannot upload: No session ID or files');
      return;
    }
    
    setUploading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });
      
      const baseURL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
      console.log('Uploading to:', `${baseURL}/api/upload/photos?session_id=${sessionId}`);
      
      const response = await axios({
        method: 'post',
        url: `${baseURL}/api/upload/photos?session_id=${sessionId}`,
        data: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      console.log('Upload response:', response);
      
      if (response.status === 200) {
        setUploadComplete(true);
        setCurrentStep(2);
      } else {
        setError(`Upload failed with status: ${response.status}`);
      }
    } catch (error) {
      console.error('Upload error:', error);
      setError(`Error uploading files: ${error.message || 'Unknown error'}`);
    } finally {
      setUploading(false);
    }
  };

  const processFiles = async (sessionId, setCurrentStep, setProcessing, setError, checkProcessingStatus) => {
    if (!sessionId) return;
    
    setProcessing(true);
    setError(null);
    
    try {
      console.log("Starting processing with session ID:", sessionId);
      
      // This matches the ProcessRequest model in the backend
      const response = await axios({
        method: 'post',
        url: `${API_BASE_URL}/api/upload/process`,
        data: { session_id: sessionId }, // Exactly matches the backend model
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      console.log("Process response:", response);
      
      if (response.status === 200) {
        setCurrentStep(3); // Move to public photos step
        
        // Start checking processing status
        checkProcessingStatus();
      } else {
        setError(`Processing failed with status: ${response.status}`);
        setProcessing(false);
      }
    } catch (error) {
      console.error("Processing error:", error);
      console.error("Error response:", error.response);
      setError(`Error processing files: ${error.message || 'Unknown error'}`);
      setProcessing(false);
    }
  };

  // The checkProcessingStatus function aligned with the backend API
  const checkProcessingStatus = async (sessionId, setProcessingComplete, setProcessing, 
    setDetectedLocations, setMemoryStats, setPublicFetchComplete, 
    setCurrentStep, setError) => {
    if (!sessionId) return;

    try {
      const response = await axios.get(`${API_BASE_URL}/api/upload/status/${sessionId}`);

        console.log("Processing status:", response.data);

      if (response.data.status === 'user_processed' || response.data.status === 'completed') {
        setProcessingComplete(true);
        setProcessing(false);

        // Use the locations_detected field from the API response
        setDetectedLocations(response.data.locations_detected || []);

        // Set memory stats from the API response
        setMemoryStats({
          user: response.data.user_photos_processed || 0,
          public: response.data.public_photos_processed || 0
        });

        // If already completed with public photos, skip to final step
        if (response.data.status === 'completed' && response.data.public_photos_processed > 0) {
          setPublicFetchComplete(true);
          setCurrentStep(4);
        }
      } else if (response.data.status === 'error') {
        setError(response.data.message);
        setProcessing(false);
      } else {
        // Still processing, check again in 2 seconds
        setTimeout(() => checkProcessingStatus(
        sessionId, 
        setProcessingComplete, 
        setProcessing, 
        setDetectedLocations, 
        setMemoryStats, 
        setPublicFetchComplete, 
        setCurrentStep, 
        setError
        ), 2000);
      }
    } catch (error) {
      console.error(`Error checking processing status: ${error}`);
      setProcessing(false);
    }
  };

  // Fetch public photos
  const fetchPublicPhotos = async () => {
    if (!sessionId || !detectedLocations.length) return;
    
    setFetchingPublic(true);
    setError(null);
    
    try {
      const response = await api.post('/api/upload/fetch-public', {
        session_id: sessionId,
        max_photos_per_location: 10,
        total_limit: 100
      });
      
      setCurrentStep(4); // Move to completion step
      
      // Start checking public fetch status
      checkPublicFetchStatus();
    } catch (error) {
      setError(`Error fetching public photos: ${error.message}`);
      setFetchingPublic(false);
    }
  };

  // Check public photo fetching status
  const checkPublicFetchStatus = async () => {
    if (!sessionId) return;
    
    try {
      const response = await api.get(`/api/upload/status/${sessionId}`);
      
      if (response.data.status === 'completed' && response.data.public_photos_processed > 0) {
        setPublicFetchComplete(true);
        setFetchingPublic(false);
        setMemoryStats({
          user: response.data.user_photos_processed || 0,
          public: response.data.public_photos_processed || 0
        });
      } else if (response.data.status === 'error') {
        setError(response.data.message);
        setFetchingPublic(false);
      } else {
        // Still processing, check again in 2 seconds
        setTimeout(checkPublicFetchStatus, 2000);
      }
    } catch (error) {
      setError(`Error checking public fetch status: ${error.message}`);
      setFetchingPublic(false);
    }
  };

  // Skip public photos fetch
  const skipPublicPhotos = () => {
    setCurrentStep(4); // Move to completion step
    setPublicFetchComplete(true);
  };

  // Continue to memory maps
  const continueToMaps = () => {
    if (onSessionCreated && sessionId) {
      onSessionCreated(sessionId);
    }
  };

  // Handle reset weight completion
  const handleResetComplete = (data) => {
    // Set a temporary success message
    if (data.status === 'success') {
      setResetMessage(`Reset ${data.updated_count} memories successfully!`);
      
      // Clear the message after 3 seconds
      setTimeout(() => {
        setResetMessage(null);
      }, 3000);
    } else {
      setError(data.message || 'Failed to reset weights');
    }
  };

  // Auto-start processing after upload completes
  useEffect(() => {
    if (uploadComplete && !processing && !processingComplete) {
      processFiles();
    }
  }, [uploadComplete]);

  // Clean up object URLs when component unmounts
  useEffect(() => {
    return () => {
      filePreviewUrls.forEach(preview => {
        if (preview.url) {
          URL.revokeObjectURL(preview.url);
        }
      });
    };
  }, [filePreviewUrls]);

  return (
    <div className="bg-white min-h-screen p-4">
      <div className="container mx-auto max-w-4xl">
        <h1 className="text-3xl font-bold mb-4 text-gray-800">Memory Cartography</h1>
        <p className="text-gray-600 mb-2">Upload your photos to create spatial memory maps</p>
        <p className="text-sm text-gray-500 mb-6">All photos are processed temporarily and not stored permanently</p>
      
        {/* Progress Steps */}
        <div className="flex justify-between mb-8">
          <div className={`flex flex-col items-center ${currentStep >= 1 ? 'text-gray-800' : 'text-gray-400'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center mb-1 ${currentStep >= 1 ? 'bg-gray-200' : 'bg-gray-100'}`}>
              <span className="text-sm">1</span>
            </div>
            <span className="text-xs">Upload</span>
          </div>
          <div className="flex-1 flex items-center">
            <div className={`h-1 w-full ${currentStep >= 2 ? 'bg-gray-400' : 'bg-gray-200'}`}></div>
          </div>
          <div className={`flex flex-col items-center ${currentStep >= 2 ? 'text-gray-800' : 'text-gray-400'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center mb-1 ${currentStep >= 2 ? 'bg-gray-200' : 'bg-gray-100'}`}>
              <span className="text-sm">2</span>
            </div>
            <span className="text-xs">Process</span>
          </div>
          <div className="flex-1 flex items-center">
            <div className={`h-1 w-full ${currentStep >= 3 ? 'bg-gray-400' : 'bg-gray-200'}`}></div>
          </div>
          <div className={`flex flex-col items-center ${currentStep >= 3 ? 'text-gray-800' : 'text-gray-400'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center mb-1 ${currentStep >= 3 ? 'bg-gray-200' : 'bg-gray-100'}`}>
              <span className="text-sm">3</span>
            </div>
            <span className="text-xs">Location</span>
          </div>
          <div className="flex-1 flex items-center">
            <div className={`h-1 w-full ${currentStep >= 4 ? 'bg-gray-400' : 'bg-gray-200'}`}></div>
          </div>
          <div className={`flex flex-col items-center ${currentStep >= 4 ? 'text-gray-800' : 'text-gray-400'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center mb-1 ${currentStep >= 4 ? 'bg-gray-200' : 'bg-gray-100'}`}>
              <span className="text-sm">4</span>
            </div>
            <span className="text-xs">Complete</span>
          </div>
        </div>
        
        {/* Reset Success Message */}
        {resetMessage && (
          <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-2 rounded mb-4 text-sm">
            {resetMessage}
          </div>
        )}
        
        {/* Error Handling */}
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            <span className="block sm:inline">{error}</span>
          </div>
        )}
        
        {/* Step 1: Upload */}
        {currentStep === 1 && (
          <>
            <div className="border-2 border-dashed rounded p-10 text-center cursor-pointer transition-colors mb-4 border-gray-300 hover:border-gray-400">
              {/* Simple file input instead of react-dropzone */}
              <div className="text-gray-400 text-4xl mb-4">📷</div>
              <p className="mb-2 text-gray-700">Click to select photos</p>
              <p className="text-sm text-gray-500 mb-4">Supports JPG, PNG, WEBP (max 20MB per image)</p>
              
              <input
                type="file"
                multiple
                accept=".jpg,.jpeg,.png,.webp"
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload"
              />
              <label htmlFor="file-upload" className="bg-gray-200 text-gray-800 px-4 py-2 rounded cursor-pointer hover:bg-gray-300">
                Select Photos
              </label>
            </div>
            
            {filePreviewUrls.length > 0 && (
              <div className="mb-6">
                <h2 className="text-lg font-semibold mb-3 text-gray-800">Selected Photos ({filePreviewUrls.length})</h2>
                
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4 mb-6">
                  {filePreviewUrls.map((preview, index) => (
                    <div key={index} className="relative rounded overflow-hidden border group">
                      <img 
                        src={preview.url}
                        alt={`Preview ${index}`}
                        className="w-full h-32 object-cover"
                      />
                      <button
                        onClick={() => removeFile(preview.file, index)}
                        className="absolute top-1 right-1 bg-red-500 text-white p-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        &times;
                      </button>
                      <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white text-xs p-1 truncate">
                        {preview.file.name}
                      </div>
                    </div>
                  ))}
                </div>
                
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={uploadFiles}
                    disabled={uploading || filePreviewUrls.length === 0}
                    className="bg-gray-200 text-gray-800 p-2 rounded disabled:opacity-50 flex-grow"
                  >
                    {uploading ? 'Uploading...' : 'Upload Photos'}
                  </button>
                  
                  {/* Include Reset Weights Button */}
                  {ResetWeightsButton && (
                    <ResetWeightsButton 
                      memoryType="user"
                      onResetComplete={handleResetComplete}
                    />
                  )}
                </div>
              </div>
            )}
          </>
        )}
        
        {/* Step 2: Processing */}
        {currentStep === 2 && (
          <div className="border rounded p-6">
            <h2 className="text-lg font-semibold mb-4 text-gray-800">Processing Your Photos</h2>
            
            <div className="flex items-center mb-6">
              <div className={`${processingComplete ? 'bg-green-100' : 'bg-gray-200'} p-2 rounded-full mr-3`}>
                {processingComplete ? (
                  <span className="text-green-600">✓</span>
                ) : (
                  <div className="animate-spin h-5 w-5 border-2 border-gray-500 border-t-transparent rounded-full"></div>
                )}
              </div>
              <div>
                <p className="font-medium text-gray-800">
                  {processingComplete ? 'Processing complete!' : 'Processing your photos...'}
                </p>
                <p className="text-sm text-gray-500">
                  Extracting features, analyzing content, and detecting locations
                </p>
              </div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded mb-6">
              <p className="text-sm text-gray-700 mb-2">
                Processing may take a few minutes depending on the number of photos.
              </p>
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div className="bg-gray-400 h-2.5 rounded-full animate-pulse"></div>
              </div>
            </div>
          </div>
        )}
        
        {/* Step 3: Locations and Public Photos */}
        {currentStep === 3 && (
          <div className="border rounded p-6">
            <h2 className="text-lg font-semibold mb-4 text-gray-800">Detected Locations</h2>
            
            {detectedLocations.length > 0 ? (
              <>
                <p className="mb-4 text-gray-700">
                  We detected the following locations in your photos. Would you like to enhance your memory maps with related public photos?
                </p>
                
                <div className="flex flex-wrap gap-2 mb-6">
                  {detectedLocations.map((location, index) => (
                    <span key={index} className="inline-block bg-gray-100 text-gray-800 px-2 py-1 rounded-full text-sm">
                      {location}
                    </span>
                  ))}
                </div>
                
                <div className="bg-gray-50 p-4 rounded mb-6">
                  <p className="text-sm text-gray-700">
                    Public photos will be temporarily downloaded to help provide context and comparison to your personal memories.
                  </p>
                </div>
                
                <div className="flex gap-2 flex-wrap">
                  <button
                    onClick={fetchPublicPhotos}
                    disabled={fetchingPublic}
                    className="bg-gray-200 text-gray-800 p-2 rounded disabled:opacity-50 flex-grow"
                  >
                    {fetchingPublic ? 'Fetching public photos...' : 'Fetch Public Photos'}
                  </button>
                  
                  <button
                    onClick={skipPublicPhotos}
                    disabled={fetchingPublic}
                    className="bg-gray-100 text-gray-700 p-2 rounded disabled:opacity-50 flex-grow"
                  >
                    Skip this step
                  </button>
                  
                  {/* Include Reset Weights Button for public memories */}
                  {ResetWeightsButton && (
                    <ResetWeightsButton 
                      memoryType="public"
                      onResetComplete={handleResetComplete}
                    />
                  )}
                </div>
              </>
            ) : (
              <>
                <p className="mb-4 text-gray-700">
                  No specific locations were detected in your photos. You can continue without public photos or try uploading more photos with clear location context.
                </p>
                
                <button
                  onClick={skipPublicPhotos}
                  className="bg-gray-200 text-gray-800 p-2 rounded w-full"
                >
                  Continue without public photos
                </button>
              </>
            )}
          </div>
        )}
        
        {/* Step 4: Complete */}
        {currentStep === 4 && (
          <div className="border rounded p-6">
            <div className="text-center mb-6">
              <div className="inline-block bg-green-100 p-4 rounded-full mb-4">
                <span className="text-green-600 text-2xl">✓</span>
              </div>
              <h2 className="text-xl font-bold mb-2 text-gray-800">Setup Complete!</h2>
              <p className="text-gray-600">Your memory maps are ready to explore</p>
            </div>
            
            <div className="bg-gray-50 p-4 rounded mb-6">
              <h3 className="font-semibold mb-2 text-gray-800">Memory Statistics</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white p-3 rounded border">
                  <p className="text-sm text-gray-500">Your Photos</p>
                  <p className="text-2xl font-bold text-gray-800">{memoryStats.user}</p>
                </div>
                <div className="bg-white p-3 rounded border">
                  <p className="text-sm text-gray-500">Public Photos</p>
                  <p className="text-2xl font-bold text-gray-800">{memoryStats.public}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded mb-6">
              <p className="text-sm text-gray-700">
                Your session and all uploaded photos will be automatically deleted after 24 hours of inactivity for privacy.
              </p>
            </div>
            
            <button
              onClick={continueToMaps}
              className="bg-gray-200 text-gray-800 p-2 rounded w-full"
            >
              Explore Your Memory Maps
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUploadZone;