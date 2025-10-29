"""
FastAPI server with WebSocket support for real-time audio streaming
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import numpy as np
import json
from typing import List
import base64
import sys

sys.path.append(str(Path(__file__).parent))
from inference.pipeline import InferencePipeline
from config import HOST, PORT, SAMPLE_RATE, CHUNK_SAMPLES, MODEL_DIR

# Initialize FastAPI app
app = FastAPI(title="SkyGuard Tactical API", version="1.0.0")

# CORS middleware for frontend (localhost + file:// for local HTML)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "null"  # Allows file:// protocol
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference pipeline
pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize inference pipeline on startup"""
    global pipeline
    print("=" * 60)
    print("SkyGuard Tactical - Starting Server")
    print("=" * 60)
    model_path = MODEL_DIR / 'best_model.pth'
    ood_path = MODEL_DIR / 'ood_detector.npz'
    pipeline = InferencePipeline(model_path=model_path, ood_path=ood_path)
    print("=" * 60)
    print("✓ Server ready at http://{}:{}".format(HOST, PORT))
    print("=" * 60)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "SkyGuard Tactical",
        "version": "1.0.0",
        "description": "Real-Time Drone Detection System"
    }


@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "pipeline_loaded": pipeline is not None,
        "sample_rate": SAMPLE_RATE,
        "chunk_size": CHUNK_SAMPLES,
        "device": pipeline.device if pipeline else "unknown"
    }


@app.post("/api/reset")
async def reset_pipeline():
    """Reset temporal smoother state"""
    if pipeline:
        pipeline.reset()
        return {"status": "reset", "message": "Pipeline state reset successfully"}
    raise HTTPException(status_code=500, detail="Pipeline not initialized")


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✓ Client connected (total: {len(self.active_connections)})")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"✗ Client disconnected (total: {len(self.active_connections)})")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections[:]:  # Create copy to avoid modification during iteration
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                self.disconnect(connection)


manager = ConnectionManager()


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming

    Expected message format:
    {
        "type": "audio",
        "data": "base64_encoded_audio",
        "sample_rate": 16000
    }
    """
    await manager.connect(websocket)

    try:
        while True:
            # Receive audio data
            data = await websocket.receive_text()
            message = json.loads(data)

            if message['type'] == 'audio':
                try:
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(message['data'])

                    # Validate audio size (max 10 seconds)
                    max_samples = SAMPLE_RATE * 10  # 10 seconds
                    if len(audio_bytes) > max_samples * 4:  # 4 bytes per float32
                        raise ValueError(f"Audio too long: max 10 seconds ({max_samples * 4} bytes)")

                    # Check 4-byte alignment for float32
                    if len(audio_bytes) % 4 != 0:
                        raise ValueError(f"Audio not 4-byte aligned (got {len(audio_bytes)} bytes)")

                    audio = np.frombuffer(audio_bytes, dtype=np.float32)

                    # Validate audio is not empty
                    if len(audio) == 0:
                        raise ValueError("Audio is empty")

                    # Process through pipeline
                    result = pipeline.process_audio(audio)

                    # Send result back to sender
                    await websocket.send_json({
                        "type": "detection",
                        "data": result
                    })

                    # Optionally broadcast to all clients
                    # await manager.broadcast({"type": "detection", "data": result})

                except Exception as e:
                    error_msg = f"Error processing audio: {str(e)}"
                    print(error_msg)
                    await websocket.send_json({
                        "type": "error",
                        "message": error_msg
                    })

            elif message['type'] == 'ping':
                await websocket.send_json({"type": "pong"})

            elif message['type'] == 'reset':
                if pipeline:
                    pipeline.reset()
                await websocket.send_json({
                    "type": "reset_ack",
                    "message": "Pipeline reset"
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.post("/api/process")
async def process_audio_file(data: dict):
    """
    Process audio file (for testing)

    Args:
        data: Dict with 'audio_base64' key containing base64 encoded audio

    Returns:
        Detection results
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        # Decode audio
        audio_base64 = data.get('audio_base64', '')
        audio_bytes = base64.b64decode(audio_base64)
        audio = np.frombuffer(audio_bytes, dtype=np.float32)

        # Process
        result = pipeline.process_audio(audio)

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
