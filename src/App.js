import React, { useRef, useEffect, useState, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import * as facemesh from "@tensorflow-models/facemesh";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [detections, setDetections] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [fps, setFps] = useState(0);
  const [isModelLoaded, setIsModelLoaded] = useState(false);

  const requestRef = useRef();
  const previousTimeRef = useRef();
  const netRef = useRef(null);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(0);

  const drawMesh = useCallback((predictions, ctx) => {
    if (predictions.length > 0) {
      predictions.forEach((prediction) => {
        const keypoints = prediction.scaledMesh;

        for (let i = 0; i < keypoints.length; i++) {
          const [x, y, z] = keypoints[i];
          ctx.beginPath();
          ctx.arc(x, y, 1, 0, 3 * Math.PI);
          ctx.fillStyle = selectedPoint === i ? "#ff0000" : "#00ff00";
          ctx.fill();
        }

        if (fps > 15) {
          for (let i = 0; i < prediction.faceInViewConfidence.length; i++) {
            const points = prediction.mesh[i];
            ctx.beginPath();
            ctx.moveTo(points[0][0], points[0][1]);
            for (let j = 1; j < points.length; j++) {
              ctx.lineTo(points[j][0], points[j][1]);
            }
            ctx.strokeStyle = "#00ff00";
            ctx.stroke();
          }
        }
      });
    }
  }, [selectedPoint, fps]);

  const recordDetection = useCallback((face) => {
    if (face && face.length > 0 && isRecording) {
      const timestamp = new Date().toLocaleTimeString();
      const newDetection = {
        timestamp,
        keypoints: face[0].scaledMesh,
        confidence: face[0].faceInViewConfidence
      };

      setDetections(prev => {
        const updated = [...prev, newDetection];
        return updated.slice(-10);
      });
    }
  }, [isRecording]);

  const detect = useCallback(async (time) => {
    if (!previousTimeRef.current) {
      previousTimeRef.current = time;
    }
    const deltaTime = time - previousTimeRef.current;

    frameCountRef.current++;
    if (time - lastFpsUpdateRef.current > 1000) {
      setFps(Math.round((frameCountRef.current * 1000) / (time - lastFpsUpdateRef.current)));
      frameCountRef.current = 0;
      lastFpsUpdateRef.current = time;
    }

    if (
      netRef.current &&
      webcamRef.current &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      try {
        const face = await netRef.current.estimateFaces(video);
        recordDetection(face);

        const ctx = canvasRef.current.getContext("2d");
        ctx.clearRect(0, 0, videoWidth, videoHeight);
        drawMesh(face, ctx);
      } catch (error) {
        console.error("Error in face detection:", error);
      }
    }

    previousTimeRef.current = time;
    requestRef.current = requestAnimationFrame(detect);
  }, [drawMesh, recordDetection]);

  useEffect(() => {
    const loadModel = async () => {
      try {
        netRef.current = await facemesh.load({
          inputResolution: { width: 640, height: 480 },
          scale: 0.8,
        });
        setIsModelLoaded(true);
        requestRef.current = requestAnimationFrame(detect);
      } catch (error) {
        console.error("Error loading model:", error);
      }
    };

    loadModel();

    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [detect]);

  const formatCoordinates = useCallback((coords) => {
    if (!coords) return "N/A";
    return `(${coords[0].toFixed(2)}, ${coords[1].toFixed(2)}, ${coords[2].toFixed(2)})`;
  }, []);

  return (
    <div className="flex flex-col md:flex-row gap-4 p-4 min-h-screen" style={{ backgroundColor: '#f3f4f6' }}>
      <div className="relative w-full md:w-1/2">
        {!isModelLoaded && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 text-white">
            Loading face detection model...
          </div>
        )}
        <Webcam
          ref={webcamRef}
          style={{
            width: 640,
            height: 480,
            borderRadius: '0.5rem'
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: 640,
            height: 480,
            borderRadius: '0.5rem'
          }}
        />
        <div style={{ marginTop: '1rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <button
            onClick={() => setIsRecording(!isRecording)}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '0.5rem',
              backgroundColor: isRecording ? '#ef4444' : '#22c55e',
              color: 'white',
              border: 'none',
              cursor: 'pointer'
            }}
          >
            {isRecording ? 'Stop Recording' : 'Start Recording'}
          </button>
          <button
            onClick={() => setDetections([])}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '0.5rem',
              backgroundColor: '#6b7280',
              color: 'white',
              border: 'none',
              cursor: 'pointer'
            }}
          >
            Clear History
          </button>
          <span style={{ fontSize: '0.875rem', color: '#4b5563' }}>FPS: {fps}</span>
        </div>
      </div>

      <div className="w-full md:w-1/2">
        <div style={{
          backgroundColor: 'white',
          borderRadius: '0.5rem',
          boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
          padding: '1.5rem'
        }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem' }}>
            Face Mesh Detections History
          </h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {detections.map((detection, idx) => (
              <div
                key={`${detection.timestamp}-${idx}`}  // Use timestamp and index as unique key
                style={{
                  padding: '1rem',
                  backgroundColor: '#f9fafb',
                  borderRadius: '0.5rem'
                }}
              >
                <h3 style={{ fontWeight: 'bold' }}>
                  Detection {idx + 1} - {detection.timestamp}
                </h3>
                <p style={{ fontSize: '0.875rem', color: '#4b5563' }}>
                  Confidence: {(detection.confidence * 100).toFixed(2)}%
                </p>
                {selectedPoint !== null && (
                  <p style={{ fontSize: '0.875rem' }}>
                    Point {selectedPoint}: {formatCoordinates(detection.keypoints[selectedPoint])}
                  </p>
                )}
              </div>
            ))}

          </div>
          <div style={{ marginTop: '1rem' }}>
            <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: '500', color: '#374151' }}>
              Select Keypoint to Track:
              <select
                value={selectedPoint || ''}
                onChange={(e) => setSelectedPoint(e.target.value ? Number(e.target.value) : null)}
                style={{
                  marginTop: '0.25rem',
                  display: 'block',
                  width: '100%',
                  padding: '0.5rem',
                  borderRadius: '0.375rem',
                  border: '1px solid #d1d5db'
                }}
              >
                <option value="">None</option>
                {[...Array(468)].map((_, i) => (
                  <option key={i} value={i}>Point {i}</option>
                ))}
              </select>
            </label>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;