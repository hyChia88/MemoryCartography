import React, { useState, useRef } from 'react';

function MemoryForm({ onImageUpload }) {
  const [previewImage, setPreviewImage] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewImage(reader.result);
      };
      reader.readAsDataURL(file);

      // Upload image
      onImageUpload(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const file = e.dataTransfer.files[0];
    if (file) {
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewImage(reader.result);
      };
      reader.readAsDataURL(file);

      // Upload image
      onImageUpload(file);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="mt-4 p-4 bg-gray-50 rounded-lg">
      <h2 className="text-xl font-semibold mb-4 text-gray-800">
        Upload Memory
      </h2>
      
      <div 
        onClick={triggerFileInput}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:bg-gray-100 transition-colors"
      >
        <input 
          type="file" 
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="image/*"
          className="hidden"
        />
        
        {previewImage ? (
          <img 
            src={previewImage} 
            alt="Preview" 
            className="max-h-48 mx-auto rounded-lg shadow-md"
          />
        ) : (
          <div>
            <p className="text-gray-500">
              Drag and drop an image or click to upload
            </p>
            <p className="text-sm text-gray-400 mt-2">
              JPG, PNG, WEBP up to 10MB
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default MemoryForm;