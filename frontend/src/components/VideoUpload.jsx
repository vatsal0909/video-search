import React, { useState } from 'react';
import { Upload, CheckCircle, Loader2, FileVideo, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import { upload_to_s3, validate_video_file } from '../utils/s3Upload';
import { getPresignedUploadUrl, completeMultipartUpload } from '../services/api';

const VideoUpload = () => {
  const [file, setFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('idle');
  const [s3Url, setS3Url] = useState('');
  const [error, setError] = useState('');

  const handle_file_select = (e) => {
    const selected_file = e.target.files[0];
    if (selected_file) {
      const validation = validate_video_file(selected_file);
      if (!validation.valid) {
        setError(validation.error);
        setFile(null);
        return;
      }
      setFile(selected_file);
      setError('');
      setUploadStatus('idle');
    }
  };

  const handle_upload = async () => {
    if (!file) return;

    try {
      setUploadStatus('uploading');
      setError('');
      setUploadProgress(0);

      // 1. Get URLs (Backend should detect large files and return multipart info)
      const presignedData = await getPresignedUploadUrl(file.name, file.size);

      // 2. Perform the upload
      const result = await upload_to_s3(file, presignedData, (progress) => {
        setUploadProgress(progress);
      });

      // 3. If Multipart, notify backend to merge parts
      if (result.type === 'multipart') {
        await completeMultipartUpload({
          uploadId: result.uploadId,
          parts: result.parts,
          fileName: file.name
        });
      }

      setS3Url(result.s3_path);
      setUploadStatus('completed');
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.message || 'Failed to upload video');
      setUploadStatus('error');
    }
  };

  const reset_form = () => {
    setFile(null);
    setUploadStatus('idle');
    setS3Url('');
    setError('');
  };

  return (
    <motion.section 
      initial={{ opacity: 0 }} 
      animate={{ opacity: 1 }} 
      className="w-full max-w-4xl mx-auto p-6"
    >
      <div className="text-center mb-8">
        <h2 className="text-4xl font-bold text-gray-800">Video Portal</h2>
        <p className="text-gray-500">Securely upload videos up to 2GB</p>
      </div>

      <div className="bg-white rounded-3xl shadow-xl p-8 border border-gray-100">
        {uploadStatus === 'idle' || uploadStatus === 'error' ? (
          <div className="space-y-6">
            <label className="group relative border-3 border-dashed border-gray-200 rounded-2xl p-12 flex flex-col items-center justify-center cursor-pointer hover:border-blue-400 hover:bg-blue-50/50 transition-all">
              <FileVideo size={48} className="text-gray-400 group-hover:text-blue-500 mb-4" />
              <span className="text-lg font-medium text-gray-600">
                {file ? file.name : "Select your video file"}
              </span>
              <span className="text-sm text-gray-400 mt-1">MP4, MOV up to 2GB</span>
              <input type="file" className="hidden" onChange={handle_file_select} accept="video/*" />
            </label>

            {error && (
              <div className="flex items-center gap-2 p-4 bg-red-50 text-red-700 rounded-xl border border-red-100">
                <AlertCircle size={20} />
                <p className="text-sm font-medium">{error}</p>
              </div>
            )}

            <button
              onClick={handle_upload}
              disabled={!file}
              className="w-full py-4 bg-blue-600 text-white rounded-2xl font-bold text-lg hover:bg-blue-700 disabled:bg-gray-200 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-200"
            >
              Start Upload
            </button>
          </div>
        ) : (
          <div className="py-12 flex flex-col items-center">
            {uploadStatus === 'uploading' ? (
              <div className="w-full space-y-6">
                <div className="flex justify-between items-end">
                  <div>
                    <h3 className="text-xl font-bold text-gray-800">Uploading...</h3>
                    <p className="text-gray-500 text-sm">Processing 2GB stream</p>
                  </div>
                  <span className="text-2xl font-black text-blue-600">{uploadProgress}%</span>
                </div>
                <div className="h-4 w-full bg-gray-100 rounded-full overflow-hidden">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${uploadProgress}%` }}
                    className="h-full bg-blue-600"
                  />
                </div>
              </div>
            ) : (
              <div className="text-center space-y-4">
                <div className="bg-green-100 p-4 rounded-full inline-block">
                  <CheckCircle size={40} className="text-green-600" />
                </div>
                <h3 className="text-2xl font-bold text-gray-800">Upload Complete!</h3>
                <p className="text-sm text-gray-500 break-all bg-gray-50 p-3 rounded-lg border">
                  Path: {s3Url}
                </p>
                <button 
                  onClick={reset_form}
                  className="mt-6 px-8 py-2 border-2 border-blue-600 text-blue-600 rounded-xl font-bold hover:bg-blue-50"
                >
                  Upload Another
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </motion.section>
  );
};

export default VideoUpload;