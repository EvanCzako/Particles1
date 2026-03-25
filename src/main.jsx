import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';

// StrictMode is omitted intentionally: it double-invokes effects in dev,
// which would spin up two WebGPU devices and two RAF loops.
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
