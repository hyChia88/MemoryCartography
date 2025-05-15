import React, { useState, useEffect, useCallback, useRef } from 'react'; // Added useRef
import axios from 'axios';
import ResetWeightsButton from './ResetWeightsButton'; // Assuming this component exists
import ProcessingComplete from './ProcessingComplete';

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
  const [currentStep, setCurrentStep] = useState(1);

  // Ref to track if initial processing has been attempted
  const processingAttemptedRef = useRef(false);

  useEffect(() => {
    const createSession = async () => {
      try {
        const response = await axios.post(`${API_BASE_URL}/api/session/create`);
        if (response.status === 200 && response.data.session_id) {
          setSessionId(response.data.session_id);
          console.log(`Session created: ${response.data.session_id}`);
        } else {
          setError(`Failed to create a session. Status: ${response.status}`);
          console.error('Session creation failed:', response);
        }
      } catch (err) {
        console.error('Session creation error:', err);
        setError(`Error creating session: ${err.response?.data?.detail || err.message || 'Unknown error'}`);
      }
    };
    createSession();
  }, []);

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(prevFiles => [...prevFiles, ...selectedFiles]);
    const newPreviewUrls = selectedFiles.map(file => ({
      file,
      url: URL.createObjectURL(file)
    }));
    setFilePreviewUrls(prevUrls => [...prevUrls, ...newPreviewUrls]);
  };

  const removeFile = (index) => {
    const fileToRemove = files[index];
    const previewUrlToRemove = filePreviewUrls[index];
    setFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
    setFilePreviewUrls(prevUrls => {
      if (previewUrlToRemove && previewUrlToRemove.url) {
        URL.revokeObjectURL(previewUrlToRemove.url);
      }
      return prevUrls.filter((_, i) => i !== index);
    });
    console.log(`Removed file: ${fileToRemove?.name}`);
  };

  const uploadFiles = async () => {
    if (!sessionId || files.length === 0) {
      console.log('Upload prerequisites not met: No session ID or no files selected.');
      setError('Please select files to upload and ensure a session is active.');
      return;
    }
    setUploading(true);
    setError(null);
    try {
      const formData = new FormData();
      files.forEach(file => formData.append('files', file));
      console.log(`Uploading ${files.length} files to: ${API_BASE_URL}/api/upload/photos?session_id=${sessionId}`);
      const response = await axios({
        method: 'post',
        url: `${API_BASE_URL}/api/upload/photos?session_id=${sessionId}`,
        data: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      console.log('Upload response:', response);
      if (response.status === 200 && response.data.status === 'success') {
        setUploadComplete(true);
        setCurrentStep(2);
        console.log('Upload successful, proceeding to processing step.');
      } else {
        setError(`Upload failed: ${response.data.message || `Status ${response.status}`}`);
        console.error('Upload failed response:', response);
      }
    } catch (err) {
      console.error('Upload error:', err);
      setError(`Error uploading files: ${err.response?.data?.detail || err.message || 'Unknown error'}`);
    } finally {
      setUploading(false);
    }
  };

  const checkProcessingStatusCallback = useCallback(async () => {
    if (!sessionId) {
      console.log('checkProcessingStatus: No session ID, stopping checks.');
      return;
    }
    console.log(`Checking processing status for session: ${sessionId}`);
    try {
      const response = await axios.get(`${API_BASE_URL}/api/upload/status/${sessionId}`);
      console.log("Processing status response:", response.data);
      const statusData = response.data;

      if (statusData.status === 'user_processed' || statusData.status === 'completed') {
        setProcessingComplete(true);
        setProcessing(false); // Processing flag off
        setDetectedLocations(statusData.locations_detected || []);
        setMemoryStats({
          user: statusData.user_photos_processed || 0,
          public: statusData.public_photos_processed || 0
        });
        if (statusData.status === 'completed' && (statusData.public_photos_processed || 0) > 0) {
          setPublicFetchComplete(true);
          setCurrentStep(4);
        } else {
          setCurrentStep(3);
        }
        console.log('Processing complete or user photos processed.');
      } else if (statusData.status === 'error' || statusData.status === 'error_db_read' || statusData.status === 'processing_error') {
        setError(`Processing error: ${statusData.message || 'Unknown processing error'}`);
        setProcessing(false); // Processing flag off on error
        console.error('Processing error reported by status endpoint:', statusData);
        // Optionally stop polling on persistent errors or implement max retries
      } else if (statusData.status === 'pending' || statusData.status === 'processing_pending_db' || statusData.status === 'processing_started') {
        console.log('Processing still in progress, will check again...');
        setTimeout(checkProcessingStatusCallback, 3000);
      } else {
        console.warn('Unknown processing status received:', statusData.status);
        setError(`Unknown processing status: ${statusData.status}. Retrying.`);
        setTimeout(checkProcessingStatusCallback, 5000);
      }
    } catch (err) {
      console.error('Error checking processing status:', err);
      setError(`Error checking status: ${err.response?.data?.detail || err.message || 'Unknown error'}. Will retry.`);
      setProcessing(false); // Turn off processing indicator on polling error
      setTimeout(checkProcessingStatusCallback, 5000); // Retry polling
    }
  }, [sessionId]); // Dependencies: only sessionId, setters are stable

  const processFiles = useCallback(async () => {
    if (!sessionId) {
      console.error('processFiles: No session ID available. Cannot start processing.');
      setError('Session ID is missing. Please try refreshing.');
      return;
    }
    setProcessing(true); // Indicate processing has started
    setError(null);
    try {
      console.log(`processFiles: Starting processing for session ID: ${sessionId}`);
      const response = await axios({
        method: 'post',
        url: `${API_BASE_URL}/api/upload/process`,
        data: { session_id: sessionId },
        headers: { 'Content-Type': 'application/json' },
      });
      console.log("Initial /process API response:", response);
      if (response.status === 200 && response.data.status === "processing_started") {
        console.log('Backend confirmed processing started. Now polling status...');
        checkProcessingStatusCallback();
      } else {
        setError(`Failed to initiate processing: ${response.data.message || `Status ${response.status}`}`);
        console.error('Failed to initiate processing:', response);
        setProcessing(false); // Reset processing if initiation failed
        processingAttemptedRef.current = false; // Allow re-attempt if initiation failed
      }
    } catch (err) {
      console.error("Error calling /process API:", err);
      setError(`Error initiating processing: ${err.response?.data?.detail || err.message || 'Unknown error'}`);
      setProcessing(false); // Reset processing on error
      processingAttemptedRef.current = false; // Allow re-attempt if initiation failed
    }
  }, [sessionId, checkProcessingStatusCallback]);

  useEffect(() => {
    // Log current state values when this effect is triggered
    console.log("useEffect (process trigger) evaluated. States:", {
      uploadComplete,
      sessionId,
      processing,
      processingComplete,
      processingAttempted: processingAttemptedRef.current
    });

    if (uploadComplete && sessionId && !processing && !processingComplete && !processingAttemptedRef.current) {
      console.log("useEffect: IF CONDITION MET. Calling processFiles.");
      processingAttemptedRef.current = true; // Set flag *before* calling processFiles
      processFiles();
    } else {
      // Log why the condition was not met
      console.log("useEffect (process trigger): IF CONDITION NOT MET.");
      if (!uploadComplete) console.log("Reason: uploadComplete is false");
      if (!sessionId) console.log("Reason: sessionId is null/false");
      if (processing) console.log("Reason: processing is true");
      if (processingComplete) console.log("Reason: processingComplete is true");
      if (processingAttemptedRef.current) console.log("Reason: processingAttemptedRef.current is true");
    }
  }, [uploadComplete, sessionId, processing, processingComplete, processFiles]); // processFiles is a stable useCallback

  const fetchPublicPhotos = async () => {
    if (!sessionId || !detectedLocations.length) {
      console.log('fetchPublicPhotos: Prerequisites not met.');
      setError('No session or detected locations to fetch public photos.');
      return;
    }
    setFetchingPublic(true);
    setError(null);
    console.log(`fetchPublicPhotos: Fetching for session ${sessionId}, locations: ${detectedLocations.join(', ')}`);
    try {
      const response = await axios.post(`${API_BASE_URL}/api/upload/fetch-public`, {
        session_id: sessionId,
        max_photos_per_location: 10,
        total_limit: 100
      });
      console.log('Fetch public photos response:', response);
      if (response.status === 200 && response.data.status === "public_fetch_started") {
        console.log('Public photo fetch initiated. Polling for completion (not fully implemented in this snippet).');
        setCurrentStep(4);
        setPublicFetchComplete(true);
      } else {
        setError(`Failed to fetch public photos: ${response.data.message || `Status ${response.status}`}`);
      }
    } catch (err) {
      console.error('Error fetching public photos:', err);
      setError(`Error fetching public photos: ${err.response?.data?.detail || err.message || 'Unknown error'}`);
    } finally {
      setFetchingPublic(false);
    }
  };

  const skipPublicPhotos = () => {
    console.log('Skipping public photo fetch.');
    setCurrentStep(4);
    setPublicFetchComplete(true);
  };

  const continueToMaps = () => {
    if (onSessionCreated && sessionId) {
      console.log(`Continuing to maps with session ID: ${sessionId}`);
      onSessionCreated(sessionId);
    } else {
      console.error('Cannot continue to maps: onSessionCreated callback or sessionId missing.');
      setError('Cannot proceed to maps. Session may not be fully ready.');
    }
  };

  const handleResetComplete = (data) => {
    if (data.status === 'success') {
      setResetMessage(`Reset ${data.updated_count} memories successfully!`);
      setTimeout(() => setResetMessage(null), 3000);
    } else {
      setError(data.message || 'Failed to reset weights');
    }
  };

  useEffect(() => {
    return () => {
      filePreviewUrls.forEach(preview => {
        if (preview.url) {
          URL.revokeObjectURL(preview.url);
          // console.log(`Revoked object URL: ${preview.url}`); // Can be noisy
        }
      });
    };
  }, []); // Keep empty to run only on unmount for cleanup

  return (
    <div className="bg-white min-h-screen p-4">
      <div className="container mx-auto max-w-4xl">
        <h1 className="text-3xl font-bold mb-4 text-gray-800">Memory Cartography</h1>
        <p className="text-gray-600 mb-2">Upload your photos to create spatial memory maps</p>
        <p className="text-sm text-gray-500 mb-6">All photos are processed temporarily and not stored permanently. Session ID: {sessionId || "Initializing..."}</p>
      
        {/* Progress Steps */}
        <div className="flex justify-between mb-8">
          {/* Step 1: Upload */}
          <div className={`flex flex-col items-center ${currentStep >= 1 ? 'text-blue-600 font-semibold' : 'text-gray-400'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center mb-1 border-2 ${currentStep >= 1 ? 'bg-blue-500 text-white border-blue-500' : 'bg-gray-100 border-gray-300'}`}>
              <span>1</span>
            </div>
            <span className="text-xs">Upload</span>
          </div>
          <div className="flex-1 flex items-center px-2">
            <div className={`h-1 w-full ${currentStep >= 2 ? 'bg-blue-500' : 'bg-gray-200'}`}></div>
          </div>
          {/* Step 2: Process */}
          <div className={`flex flex-col items-center ${currentStep >= 2 ? 'text-blue-600 font-semibold' : 'text-gray-400'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center mb-1 border-2 ${currentStep >= 2 ? 'bg-blue-500 text-white border-blue-500' : 'bg-gray-100 border-gray-300'}`}>
              <span>2</span>
            </div>
            <span className="text-xs">Process</span>
          </div>
          <div className="flex-1 flex items-center px-2">
            <div className={`h-1 w-full ${currentStep >= 3 ? 'bg-blue-500' : 'bg-gray-200'}`}></div>
          </div>
          {/* Step 3: Location/Public */}
          <div className={`flex flex-col items-center ${currentStep >= 3 ? 'text-blue-600 font-semibold' : 'text-gray-400'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center mb-1 border-2 ${currentStep >= 3 ? 'bg-blue-500 text-white border-blue-500' : 'bg-gray-100 border-gray-300'}`}>
              <span>3</span>
            </div>
            <span className="text-xs">Enhance</span>
          </div>
          <div className="flex-1 flex items-center px-2">
            <div className={`h-1 w-full ${currentStep >= 4 ? 'bg-blue-500' : 'bg-gray-200'}`}></div>
          </div>
          {/* Step 4: Complete */}
          <div className={`flex flex-col items-center ${currentStep >= 4 ? 'text-blue-600 font-semibold' : 'text-gray-400'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center mb-1 border-2 ${currentStep >= 4 ? 'bg-blue-500 text-white border-blue-500' : 'bg-gray-100 border-gray-300'}`}>
              <span>4</span>
            </div>
            <span className="text-xs">Complete</span>
          </div>
        </div>
        
        {resetMessage && (
          <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-2 rounded mb-4 text-sm">
            {resetMessage}
          </div>
        )}
        
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            <strong className="font-bold">Error: </strong>
            <span className="block sm:inline">{error}</span>
          </div>
        )}
        
        {/* Step 1: Upload UI */}
        {currentStep === 1 && (
          <>
            <div className="border-2 border-dashed rounded p-10 text-center cursor-pointer transition-colors mb-4 border-gray-300 hover:border-blue-400">
              <div className="text-gray-400 text-4xl mb-4">📷</div>
              <p className="mb-2 text-gray-700">Click to select photos or drag & drop</p>
              <p className="text-sm text-gray-500 mb-4">Supports JPG, PNG, WEBP (max 20MB per image recommended)</p>
              
              <input
                type="file"
                multiple
                accept=".jpg,.jpeg,.png,.webp"
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload"
              />
              <label htmlFor="file-upload" className="bg-blue-500 text-white px-6 py-2 rounded cursor-pointer hover:bg-blue-600 transition-colors">
                Select Photos
              </label>
            </div>
            
            {filePreviewUrls.length > 0 && (
              <div className="mb-6">
                <h2 className="text-lg font-semibold mb-3 text-gray-800">Selected Photos ({filePreviewUrls.length})</h2>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 mb-6">
                  {filePreviewUrls.map((preview, index) => (
                    <div key={index} className="relative rounded overflow-hidden border group shadow-sm">
                      <img 
                        src={preview.url}
                        alt={`Preview ${preview.file.name}`}
                        className="w-full h-32 object-cover"
                      />
                      <button
                        onClick={() => removeFile(index)}
                        className="absolute top-1 right-1 bg-red-600 text-white w-6 h-6 rounded-full flex items-center justify-center text-xs opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-700"
                        aria-label="Remove image"
                      >
                        &times;
                      </button>
                      <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-60 text-white text-xs p-1 truncate" title={preview.file.name}>
                        {preview.file.name}
                      </div>
                    </div>
                  ))}
                </div>
                
                <div className="flex flex-col sm:flex-row flex-wrap gap-2">
                  <button
                    onClick={uploadFiles}
                    disabled={uploading || filePreviewUrls.length === 0 || !sessionId}
                    className="bg-green-500 text-white px-6 py-2 rounded disabled:opacity-50 hover:bg-green-600 transition-colors flex-grow"
                  >
                    {uploading ? 'Uploading...' : `Upload ${filePreviewUrls.length} Photo(s)`}
                  </button>
                  
                  {ResetWeightsButton && sessionId && (
                    <ResetWeightsButton 
                      sessionId={sessionId} 
                      memoryType="user" 
                      onResetComplete={handleResetComplete}
                    />
                  )}
                </div>
              </div>
            )}
          </>
        )}
        
        {/* Step 2: Processing UI */}
        {currentStep === 2 && (
          <div className="border rounded p-6 shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Processing Your Photos</h2>
            <div className="flex items-center mb-6">
              <div className={`p-2 rounded-full mr-3 ${processingComplete ? 'bg-green-100' : 'bg-blue-100'}`}>
                {processingComplete ? (
                  <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                ) : (
                  <div className="animate-spin h-6 w-6 border-4 border-blue-500 border-t-transparent rounded-full"></div>
                )}
              </div>
              <div>
                <p className="font-medium text-gray-800">
                  {processingComplete ? 'Processing complete!' : (processing ? 'Processing your photos...' : 'Initiating processing...')}
                </p>
                <p className="text-sm text-gray-500">
                  Extracting features, analyzing content, and detecting locations. This may take a few moments.
                </p>
              </div>
            </div>
            
            {!processingComplete && processing && (
              <div className="bg-gray-50 p-4 rounded mb-6">
                <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
                  <div className="bg-blue-500 h-2.5 rounded-full animate-pulse" style={{ width: '100%' }}></div>
                </div>
                 <p className="text-xs text-gray-500 mt-2 text-center">Please wait, this step can take some time...</p>
              </div>
            )}
             {/* Button to manually re-check status or re-initiate if stuck (optional) */}
            {!processingComplete && !processing && processingAttemptedRef.current && (
              <div className="mt-4 text-center">
                <p className="text-sm text-gray-600 mb-2">Processing seems to be taking a while or encountered an issue.</p>
                <button
                  onClick={() => {
                    console.log("Manual retry of checkProcessingStatusCallback initiated.");
                    // Reset error before retrying if you want
                    // setError(null); 
                    checkProcessingStatusCallback();
                  }}
                  className="bg-yellow-500 text-white px-4 py-2 rounded hover:bg-yellow-600 transition-colors text-sm"
                >
                  Retry Status Check
                </button>
              </div>
            )}
          </div>
        )}
        
        {/* Step 3: Locations and Public Photos UI */}
        {currentStep === 3 && (
          <div className="border rounded p-6 shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Enhance Your Map</h2>
            {detectedLocations.length > 0 ? (
              <>
                <p className="mb-2 text-gray-700">
                  We found photos related to these locations:
                </p>
                <div className="flex flex-wrap gap-2 mb-6">
                  {detectedLocations.map((location, index) => (
                    <span key={index} className="inline-block bg-indigo-100 text-indigo-800 px-3 py-1 rounded-full text-sm font-medium">
                      {location}
                    </span>
                  ))}
                </div>
                <p className="mb-4 text-gray-700">
                  Would you like to add related public photos to enrich your memory map?
                </p>
                <div className="bg-gray-50 p-4 rounded mb-6 text-sm text-gray-600">
                  Public photos are temporarily sourced to provide broader context and will not be stored with your personal data.
                </div>
                <div className="flex gap-3 flex-wrap">
                  <button
                    onClick={fetchPublicPhotos}
                    disabled={fetchingPublic}
                    className="bg-indigo-500 text-white px-6 py-2 rounded disabled:opacity-50 hover:bg-indigo-600 transition-colors flex-grow"
                  >
                    {fetchingPublic ? 'Fetching...' : 'Fetch Public Photos'}
                  </button>
                  <button
                    onClick={skipPublicPhotos}
                    disabled={fetchingPublic}
                    className="bg-gray-200 text-gray-700 px-6 py-2 rounded hover:bg-gray-300 transition-colors flex-grow"
                  >
                    Skip
                  </button>
                </div>
              </>
            ) : (
              <>
                <p className="mb-4 text-gray-700">
                  No specific new locations were prominently detected in your latest photos, or processing is still finalizing.
                  You can proceed to view your map.
                </p>
                <button
                  onClick={skipPublicPhotos}
                  className="bg-blue-500 text-white px-6 py-2 rounded w-full hover:bg-blue-600 transition-colors"
                >
                  Continue to Map
                </button>
              </>
            )}
          </div>
        )}
        
        {/* Step 4: Complete UI */}
        {currentStep === 4 && (
          <div className="border rounded p-6 shadow-md text-center">
            <div className="inline-block bg-green-100 p-3 rounded-full mb-4">
              <svg className="w-10 h-10 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
            </div>
            <h2 className="text-2xl font-bold mb-2 text-gray-800">Setup Complete!</h2>
            <p className="text-gray-600 mb-6">Your memory map is ready to explore.</p>
            
            <div className="bg-gray-50 p-4 rounded mb-6 text-left">
              <h3 className="font-semibold mb-3 text-gray-800">Memory Statistics:</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="bg-white p-4 rounded border shadow-sm">
                  <p className="text-sm text-gray-500 mb-1">Your Processed Photos</p>
                  <p className="text-3xl font-bold text-blue-600">{memoryStats.user}</p>
                </div>
                <div className="bg-white p-4 rounded border shadow-sm">
                  <p className="text-sm text-gray-500 mb-1">Public Context Photos</p>
                  <p className="text-3xl font-bold text-indigo-600">{memoryStats.public}</p>
                </div>
              </div>
            </div>
            
            <button
              onClick={continueToMaps}
              className="bg-green-500 text-white px-8 py-3 rounded w-full sm:w-auto hover:bg-green-600 transition-colors text-lg font-semibold"
            >
              Explore Your Memory Maps
            </button>
             <p className="text-xs text-gray-500 mt-4">
                Session ID: {sessionId}. This session will be cleared after a period of inactivity.
              </p>
            <div>
              <ProcessingComplete sessionId={sessionId} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUploadZone;
