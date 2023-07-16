import React, { useState, useRef } from 'react';
import axios from 'axios';
import { BrowserRouter as Router, Switch, Route, Link } from 'react-router-dom';
import './styles.css';

function App() {
  const [output, setOutput] = useState('');
  const [confidence, setConfidence] = useState('');
  const [error, setError] = useState('');
  const [videoSrc, setVideoSrc] = useState('');

  const videoRef = useRef(null);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!['video/mp4', 'video/avi', 'video/quicktime'].includes(file.type)) {
      setError('Invalid file format. Only .mp4, .avi, and .mov files are allowed');
      return;
    }

    const formData = new FormData();
    formData.append('video', file);

    axios.post('http://127.0.0.1:5000/Detect', formData)
      .then((response) => {
        const { data } = response;
        if (data.error) {
          setError(data.error);
          setOutput('');
          setConfidence('');
          setVideoSrc('');
        } else {
          setOutput(data.output);
          setConfidence(data.confidence);
          setError('');
          setVideoSrc(URL.createObjectURL(file));
        }
      })
      .catch((error) => {
        console.error(error);
        setError('An error occurred while processing the video');
        setOutput('');
        setConfidence('');
        setVideoSrc('');
      });
  };

  return (
    <div className="container">
      <h1>Deepfake Video Detection</h1>
      <input
        className="file-input"
        type="file"
        accept="video/mp4, video/avi, video/quicktime"
        onChange={handleFileUpload}
      />
      {error && <p className="error">{error}</p>}
      {videoSrc && (
        <div className="video-container">
          <video ref={videoRef} className="centered-video" controls>
            <source src={videoSrc} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>
      )}
      {output !== '' && (
        <div className="result-container">
          <h2 className="result-title">Result:</h2>
          <p className="result-content">{output}</p>
          <p className="result-content">Confidence: {confidence}</p>
        </div>
      )}
    </div>
  );
}

export default App;
