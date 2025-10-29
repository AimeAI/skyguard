PART 4: Remaining Frontend Components & DocumentationFRONTEND: frontend/components/Spectrogram.tsx
typescript/**
 * Real-time spectrogram visualization
 */
import React, { useEffect, useRef } from 'react';

interface SpectrogramProps {
  audioData?: Float32Array;
  sampleRate?: number;
  className?: string;
}

export const Spectrogram: React.FC<SpectrogramProps> = ({
  audioData,
  sampleRate = 16000,
  className = ''
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const spectrogramDataRef = useRef<number[][]>([]);
  const maxColumns = 200; // Keep last 200 time steps
  
  useEffect(() => {
    if (!audioData || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Compute FFT (simplified - in production use proper FFT library)
    const fftSize = 512;
    const fftData = computeFFT(audioData.slice(0, fftSize));
    
    // Add new column to spectrogram
    spectrogramDataRef.current.push(fftData);
    if (spectrogramDataRef.current.length > maxColumns) {
      spectrogramDataRef.current.shift();
    }
    
    // Render spectrogram
    renderSpectrogram(ctx, canvas.width, canvas.height, spectrogramDataRef.current);
    
  }, [audioData]);
  
  const computeFFT = (signal: Float32Array): number[] => {
    // Simplified FFT - in production, use a proper library
    const n = signal.length;
    const result: number[] = [];
    
    for (let k = 0; k < n / 2; k++) {
      let real = 0;
      let imag = 0;
      
      for (let i = 0; i < n; i++) {
        const angle = (2 * Math.PI * k * i) / n;
        real += signal[i] * Math.cos(angle);
        imag -= signal[i] * Math.sin(angle);
      }
      
      const magnitude = Math.sqrt(real * real + imag * imag);
      result.push(magnitude);
    }
    
    return result;
  };
  
  const renderSpectrogram = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    data: number[][]
  ) => {
    // Clear canvas
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);
    
    if (data.length === 0) return;
    
    const numFreqs = data[0].length;
    const colWidth = width / maxColumns;
    const rowHeight = height / numFreqs;
    
    // Find global min/max for normalization
    let maxVal = 0;
    data.forEach(col => {
      col.forEach(val => {
        maxVal = Math.max(maxVal, val);
      });
    });
    
    // Draw spectrogram
    data.forEach((column, colIdx) => {
      column.forEach((value, rowIdx) => {
        const normalized = value / maxVal;
        const color = getColorForValue(normalized);
        
        ctx.fillStyle = color;
        ctx.fillRect(
          colIdx * colWidth,
          height - (rowIdx + 1) * rowHeight,
          colWidth + 1,
          rowHeight + 1
        );
      });
    });
    
    // Draw frequency labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px monospace';
    ctx.textAlign = 'right';
    
    const freqLabels = ['0Hz', '2kHz', '4kHz', '6kHz', '8kHz'];
    freqLabels.forEach((label, idx) => {
      const y = height - (idx * height / (freqLabels.length - 1));
      ctx.fillText(label, width - 5, y);
    });
  };
  
  const getColorForValue = (normalized: number): string => {
    // Jet colormap (blue -> cyan -> yellow -> red)
    if (normalized < 0.25) {
      const t = normalized / 0.25;
      return `rgb(0, ${Math.floor(t * 255)}, 255)`;
    } else if (normalized < 0.5) {
      const t = (normalized - 0.25) / 0.25;
      return `rgb(0, 255, ${Math.floor((1 - t) * 255)})`;
    } else if (normalized < 0.75) {
      const t = (normalized - 0.5) / 0.25;
      return `rgb(${Math.floor(t * 255)}, 255, 0)`;
    } else {
      const t = (normalized - 0.75) / 0.25;
      return `rgb(255, ${Math.floor((1 - t) * 255)}, 0)`;
    }
  };
  
  return (
    <div className={`bg-slate-800 rounded-lg p-4 shadow-xl ${className}`}>
      <h3 className="text-lg font-bold text-white mb-3">Real-Time Spectrogram</h3>
      <canvas
        ref={canvasRef}
        width={800}
        height={300}
        className="w-full rounded border border-slate-700"
      />
      <div className="text-xs text-slate-400 mt-2 text-center">
        Frequency (Hz) vs Time
      </div>
    </div>
  );
};FRONTEND: frontend/components/EventLog.tsx
typescript/**
 * Event log with timestamps and export capability
 */
import React, { useState, useEffect } from 'react';
import { Download, Trash2 } from 'lucide-react';

interface LogEntry {
  timestamp: Date;
  className: string;
  confidence: number;
  isOOD: boolean;
  latencyMs: number;
}

interface EventLogProps {
  maxEntries?: number;
  className?: string;
}

export const EventLog: React.FC<EventLogProps> = ({
  maxEntries = 100,
  className = ''
}) => {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  
  const addEntry = (entry: Omit<LogEntry, 'timestamp'>) => {
    setEntries(prev => {
      const newEntry = { ...entry, timestamp: new Date() };
      const updated = [newEntry, ...prev];
      return updated.slice(0, maxEntries);
    });
  };
  
  const clearLog = () => {
    if (confirm('Clear all log entries?')) {
      setEntries([]);
    }
  };
  
  const exportLog = () => {
    const csv = [
      'Timestamp,Class,Confidence,OOD,Latency(ms)',
      ...entries.map(e => 
        `${e.timestamp.toISOString()},${e.className},${e.confidence},${e.isOOD},${e.latencyMs}`
      )
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `skyguard-log-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  // Expose addEntry to parent (via ref or context)
  useEffect(() => {
    (window as any).addLogEntry = addEntry;
    return () => {
      delete (window as any).addLogEntry;
    };
  }, []);
  
  return (
    <div className={`bg-slate-800 rounded-lg shadow-xl ${className}`}>
      <div className="flex items-center justify-between p-4 border-b border-slate-700">
        <h3 className="text-lg font-bold text-white">Event Log</h3>
        <div className="flex gap-2">
          <button
            onClick={exportLog}
            disabled={entries.length === 0}
            className="p-2 text-slate-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
            title="Export CSV"
          >
            <Download size={20} />
          </button>
          <button
            onClick={clearLog}
            disabled={entries.length === 0}
            className="p-2 text-slate-400 hover:text-red-400 disabled:opacity-30 disabled:cursor-not-allowed"
            title="Clear Log"
          >
            <Trash2 size={20} />
          </button>
        </div>
      </div>
      
      <div className="overflow-y-auto max-h-96">
        {entries.length === 0 ? (
          <div className="p-8 text-center text-slate-500">
            No events logged yet
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-slate-900 text-slate-400">
              <tr>
                <th className="px-4 py-2 text-left">Time</th>
                <th className="px-4 py-2 text-left">Classification</th>
                <th className="px-4 py-2 text-right">Confidence</th>
                <th className="px-4 py-2 text-right">Latency</th>
              </tr>
            </thead>
            <tbody className="text-slate-300">
              {entries.map((entry, idx) => (
                <tr
                  key={idx}
                  className="border-t border-slate-700 hover:bg-slate-750"
                >
                  <td className="px-4 py-2 font-mono text-xs">
                    {entry.timestamp.toLocaleTimeString()}
                  </td>
                  <td className="px-4 py-2">
                    <span className={`
                      ${entry.className === 'Non-Drone' ? 'text-slate-400' : 'text-white font-semibold'}
                    `}>
                      {entry.className}
                    </span>
                    {entry.isOOD && (
                      <span className="ml-2 px-2 py-0.5 bg-purple-600 text-white text-xs rounded">
                        Unknown
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-2 text-right font-mono">
                    {(entry.confidence * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-xs">
                    {entry.latencyMs.toFixed(0)}ms
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      
      <div className="p-3 border-t border-slate-700 text-xs text-slate-500 text-center">
        {entries.length} / {maxEntries} entries
      </div>
    </div>
  );
};FRONTEND: frontend/components/MetricsDashboard.tsx
typescript/**
 * Metrics dashboard with confusion matrix and performance stats
 */
import React, { useState, useEffect } from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface MetricsDashboardProps {
  className?: string;
}

export const MetricsDashboard: React.FC<MetricsDashboardProps> = ({
  className = ''
}) => {
  const [metrics, setMetrics] = useState({
    totalDetections: 0,
    avgConfidence: 0,
    avgLatency: 0,
    falsePositives: 0,
    oodDetections: 0
  });
  
  const [latencyHistory, setLatencyHistory] = useState<number[]>([]);
  
  const updateMetrics = (latency: number, confidence: number, isOOD: boolean) => {
    setMetrics(prev => ({
      totalDetections: prev.totalDetections + 1,
      avgConfidence: (prev.avgConfidence * prev.totalDetections + confidence) / (prev.totalDetections + 1),
      avgLatency: (prev.avgLatency * prev.totalDetections + latency) / (prev.totalDetections + 1),
      falsePositives: prev.falsePositives,
      oodDetections: prev.oodDetections + (isOOD ? 1 : 0)
    }));
    
    setLatencyHistory(prev => [...prev.slice(-49), latency]);
  };
  
  // Expose to parent
  useEffect(() => {
    (window as any).updateMetrics = updateMetrics;
    return () => {
      delete (window as any).updateMetrics;
    };
  }, []);
  
  const latencyChartData = {
    labels: latencyHistory.map((_, i) => i.toString()),
    datasets: [
      {
        label: 'Latency (ms)',
        data: latencyHistory,
        backgroundColor: latencyHistory.map(l => 
          l < 200 ? 'rgba(34, 197, 94, 0.7)' : 'rgba(251, 191, 36, 0.7)'
        ),
        borderColor: latencyHistory.map(l => 
          l < 200 ? 'rgba(34, 197, 94, 1)' : 'rgba(251, 191, 36, 1)'
        ),
        borderWidth: 1
      }
    ]
  };
  
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: 'Processing Latency Over Time',
        color: '#e2e8f0'
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 300,
        grid: {
          color: '#334155'
        },
        ticks: {
          color: '#94a3b8'
        }
      },
      x: {
        display: false
      }
    }
  };
  
  return (
    <div className={`bg-slate-800 rounded-lg p-6 shadow-xl ${className}`}>
      <h3 className="text-lg font-bold text-white mb-6">Performance Metrics</h3>
      
      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-slate-900 rounded-lg p-4">
          <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">
            Total Detections
          </div>
          <div className="text-2xl font-bold text-white">
            {metrics.totalDetections}
          </div>
        </div>
        
        <div className="bg-slate-900 rounded-lg p-4">
          <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">
            Avg Confidence
          </div>
          <div className="text-2xl font-bold text-green-400">
            {(metrics.avgConfidence * 100).toFixed(1)}%
          </div>
        </div>
        
        <div className="bg-slate-900 rounded-lg p-4">
          <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">
            Avg Latency
          </div>
          <div className={`text-2xl font-bold ${
            metrics.avgLatency < 200 ? 'text-green-400' : 'text-yellow-400'
          }`}>
            {metrics.avgLatency.toFixed(0)}ms
          </div>
        </div>
        
        <div className="bg-slate-900 rounded-lg p-4">
          <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">
            Unknown Drones
          </div>
          <div className="text-2xl font-bold text-purple-400">
            {metrics.oodDetections}
          </div>
        </div>
      </div>
      
      {/* Latency Chart */}
      <div className="h-64">
        <Bar data={latencyChartData} options={chartOptions} />
      </div>
      
      {/* Model Info */}
      <div className="mt-6 p-4 bg-slate-900 rounded-lg">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-slate-400">Model:</span>
            <span className="ml-2 text-white font-mono">YAMNet + Fine-tuned</span>
          </div>
          <div>
            <span className="text-slate-400">Classes:</span>
            <span className="ml-2 text-white font-mono">11 (10 drones + non-drone)</span>
          </div>
          <div>
            <span className="text-slate-400">Sample Rate:</span>
            <span className="ml-2 text-white font-mono">16 kHz</span>
          </div>
          <div>
            <span className="text-slate-400">Target Latency:</span>
            <span className="ml-2 text-white font-mono">&lt; 200ms</span>
          </div>
        </div>
      </div>
    </div>
  );
};FRONTEND: frontend/components/TacticalMap.tsx
typescript/**
 * Tactical map with bearing visualization
 */
import React, { useEffect, useRef } from 'react';
import { MapPin, Radio } from 'lucide-react';

interface Bearing {
  angle: number; // degrees from north
  distance: number; // relative distance (0-1)
  confidence: number;
}

interface TacticalMapProps {
  centerLat?: number;
  centerLon?: number;
  bearings?: Bearing[];
  className?: string;
}

export const TacticalMap: React.FC<TacticalMapProps> = ({
  centerLat = 45.4215, // Ottawa coordinates
  centerLon = -75.6972,
  bearings = [],
  className = ''
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    renderMap(ctx, canvas.width, canvas.height, bearings);
  }, [bearings]);
  
  const renderMap = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    bearings: Bearing[]
  ) => {
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 2 - 40;
    
    // Clear canvas
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);
    
    // Draw compass rose
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    ctx.stroke();
    
    // Draw range rings
    [0.33, 0.67, 1.0].forEach((scale, idx) => {
      ctx.strokeStyle = idx === 2 ? '#475569' : '#1e293b';
      ctx.lineWidth = idx === 2 ? 2 : 1;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius * scale, 0, 2 * Math.PI);
      ctx.stroke();
    });
    
    // Draw cardinal directions
    ctx.fillStyle = '#94a3b8';
    ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    const directions = [
      { label: 'N', angle: 0 },
      { label: 'E', angle: 90 },
      { label: 'S', angle: 180 },
      { label: 'W', angle: 270 }
    ];
    
    directions.forEach(({ label, angle }) => {
      const rad = (angle - 90) * Math.PI / 180;
      const x = centerX + Math.cos(rad) * (radius + 25);
      const y = centerY + Math.sin(rad) * (radius + 25);
      ctx.fillText(label, x, y);
    });
    
    // Draw sensor position (center)
    ctx.fillStyle = '#3b82f6';
    ctx.beginPath();
    ctx.arc(centerX, centerY, 8, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw bearings
    bearings.forEach(bearing => {
      const angle = (bearing.angle - 90) * Math.PI / 180;
      const endRadius = radius * bearing.distance;
      const endX = centerX + Math.cos(angle) * endRadius;
      const endY = centerY + Math.sin(angle) * endRadius;
      
      // Bearing line
      ctx.strokeStyle = `rgba(239, 68, 68, ${bearing.confidence})`;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(endX, endY);
      ctx.stroke();
      
      // Confidence cone
      ctx.fillStyle = `rgba(239, 68, 68, ${bearing.confidence * 0.2})`;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      const coneAngle = 15 * Math.PI / 180;
      ctx.arc(centerX, centerY, endRadius, angle - coneAngle, angle + coneAngle);
      ctx.closePath();
      ctx.fill();
      
      // Target marker
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(endX, endY, 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#fca5a5';
      ctx.lineWidth = 2;
      ctx.stroke();
    });
    
    // Legend
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`Lat: ${centerLat.toFixed(4)}°`, 10, height - 35);
    ctx.fillText(`Lon: ${centerLon.toFixed(4)}°`, 10, height - 20);
    ctx.fillText(`Detections: ${bearings.length}`, 10, height - 5);
  };
  
  return (
    <div className={`bg-slate-800 rounded-lg p-4 shadow-xl ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-bold text-white">Tactical Map</h3>
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <Radio size={16} />
          <span>Multi-Sensor Mode</span>
        </div>
      </div>
      
      <canvas
        ref={canvasRef}
        width={600}
        height={600}
        className="w-full rounded border border-slate-700"
      />
      
      <div className="mt-3 p-3 bg-slate-900 rounded text-xs text-slate-400">
        <div className="flex items-center gap-2 mb-1">
          <MapPin size={14} className="text-blue-400" />
          <span>Blue marker: Sensor position</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-red-500 rounded-full"></div>
          <span>Red cone: Estimated bearing to drone</span>
        </div>
      </div>
    </div>
  );
};FRONTEND: frontend/app/page.tsx
typescript/**
 * Main dashboard page
 */
'use client';

import React, { useState, useEffect, useRef } from 'react';
import { ThreatStatus } from '@/components/ThreatStatus';
import { ClassificationDisplay } from '@/components/ClassificationDisplay';
import { Spectrogram } from '@/components/Spectrogram';
import { EventLog } from '@/components/EventLog';
import { MetricsDashboard } from '@/components/MetricsDashboard';
import { TacticalMap } from '@/components/TacticalMap';
import { AudioWebSocket, DetectionResult } from '@/lib/websocket';
import { Mic, MicOff, Activity } from 'lucide-react';

export default function Home() {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [currentResult, setCurrentResult] = useState<DetectionResult | null>(null);
  const [audioData, setAudioData] = useState<Float32Array | null>(null);
  
  const wsRef = useRef<AudioWebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  
  useEffect(() => {
    // Initialize WebSocket
    wsRef.current = new AudioWebSocket('ws://localhost:8000/ws/audio');
    
    wsRef.current.connect(
      (result) => {
        setCurrentResult(result);
        setIsConnected(true);
        
        // Update metrics and log
        if (typeof window !== 'undefined') {
          (window as any).updateMetrics?.(
            result.latency_ms,
            result.confidence,
            result.is_ood
          );
          
          if (result.detected) {
            (window as any).addLogEntry?.({
              className: result.class_name,
              confidence: result.confidence,
              isOOD: result.is_ood,
              latencyMs: result.latency_ms
            });
          }
        }
      },
      (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      },
      (event) => {
        setIsConnected(false);
      }
    );
    
    return () => {
      wsRef.current?.disconnect();
      stopRecording();
    };
  }, []);
  
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(16384, 1, 1);
      processorRef.current = processor;
      
      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        const audioChunk = new Float32Array(inputData);
        
        // Send to server
        wsRef.current?.sendAudio(audioChunk, 16000);
        
        // Update spectrogram
        setAudioData(audioChunk);
      };
      
      source.connect(processor);
      processor.connect(audioContext.destination);
      
      setIsRecording(true);
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Could not access microphone. Please grant permission.');
    }
  };
  
  const stopRecording = () => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    setIsRecording(false);
    setAudioData(null);
  };
  
  return (
    <div className="min-h-screen bg-slate-900 text-white">
      {/* Header */}
      <header className="bg-slate-800 border-b border-slate-700 shadow-lg">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Activity size={32} className="text-blue-400" />
              <div>
                <h1 className="text-2xl font-bold">SkyGuard Tactical</h1>
                <p className="text-sm text-slate-400">Real-Time Drone Detection System</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${
                isConnected ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'
                }`} />
                <span className="text-sm font-medium">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              
              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
                  isRecording
                    ? 'bg-red-600 hover:bg-red-700 text-white'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
              >
                {isRecording ? (
                  <>
                    <MicOff size={20} />
                    Stop Listening
                  </>
                ) : (
                  <>
                    <Mic size={20} />
                    Start Listening
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Primary Status */}
          <div className="lg:col-span-2 space-y-6">
            <ThreatStatus
              detected={currentResult?.detected || false}
              confidence={currentResult?.confidence || 0}
            />
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <ClassificationDisplay
                className={currentResult?.class_name || 'Non-Drone'}
                confidence={currentResult?.confidence || 0}
                isOOD={currentResult?.is_ood || false}
                latencyMs={currentResult?.latency_ms || 0}
              />
              
              <TacticalMap
                bearings={currentResult?.detected ? [
                  {
                    angle: 45,
                    distance: 0.7,
                    confidence: currentResult.confidence
                  }
                ] : []}
              />
            </div>
            
            <Spectrogram audioData={audioData || undefined} />
          </div>
          
          {/* Right Column - Logs & Metrics */}
          <div className="space-y-6">
            <EventLog maxEntries={50} />
            <MetricsDashboard />
          </div>
        </div>
      </main>
      
      {/* Footer */}
      <footer className="bg-slate-800 border-t border-slate-700 mt-12">
        <div className="container mx-auto px-6 py-4 text-center text-sm text-slate-400">
          <p>SkyGuard Tactical v1.0 | "Shazam for Drones" Hackathon | November 2025</p>
        </div>
      </footer>
    </div>
  );
}