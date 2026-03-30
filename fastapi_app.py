"""
Mind Mate Emotion Detection API - FastAPI Version
مع WebSocket للـ Real-time Detection
"""

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List #
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from emotion import init as init_emotion_classifier, detect_emotion, emotions as EMOTIONS
import base64  #
import logging # يسجل ال logs 
from datetime import datetime
import traceback  # يطبع تفاصيل الخطا 
import asyncio # 
from fastapi import Form, Header
import jwt
# إعداد الـ logging




logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(
    title="Mind Mate Emotion Detection API",
    description="Face Detection & Emotion Recognition API with WebSocket support",
    version="2.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # في الإنتاج، حدد الـ origins المسموحة
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
    YOLO_MODEL_PATH = './yolov8n-face.pt'
    EMOTION_MODEL_PATH = './repvgg.pth'
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4 # 

# Pydantic Models
class Base64ImageRequest(BaseModel):
    image_base64: str

class EmotionResult(BaseModel):
    face_id: int
    emotion: str
    bbox: List[int]
    confidence: float

class DetectionResponse(BaseModel):
    success: bool
    faces_count: int
    results: List[EmotionResult]
    processing_time: float
    message: Optional[str] = None

class EmotionResultWithProbs(BaseModel):
    face_id: int
    emotion: str
    bbox: List[int]
    confidence: float
    emotion_probabilities: dict

class DetectionResponseWithProbs(BaseModel):
    success: bool
    faces_count: int
    results: List[EmotionResultWithProbs]
    processing_time: float
    message: Optional[str] = None

# Face Detector Class
class FaceDetector:
    def __init__(self, model_path: str, device: str, conf_threshold: float = 0.5, iou_threshold: float = 0.4):
        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model.to(self.device)
        logger.info(f"Face detector initialized on {device}")

    def detect_faces(self, image: np.ndarray):
        try:
            results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': int(cls)
                    })
            return detections
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            raise

# Global variables
detector: Optional[FaceDetector] = None
device: Optional[torch.device] = None

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

manager = ConnectionManager()

# Helper Functions
def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise

def process_image(image: np.ndarray, return_probs: bool = False):
    """Process image and return emotion detection results"""
    start_time = datetime.now()
    
    # Face detection
    detections = detector.detect_faces(image)
    logger.info(f"Detected {len(detections)} faces")
    
    # Emotion classification
    results = []
    if len(detections) > 0:
        face_crops = []
        valid_detections = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            # Validate bbox
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid bbox: {det['bbox']}")
                continue
            
            face_crop = image[y1:y2, x1:x2]
            face_crops.append(face_crop)
            valid_detections.append(det)
        
        # Get emotions
        if len(face_crops) > 0:
            emotions_results = detect_emotion(face_crops, conf=False, return_probs=return_probs)
            
            for i, det in enumerate(valid_detections):
                if i < len(emotions_results):
                    if return_probs:
                        label, idx, probs = emotions_results[i]
                        emotion_probs = {
                            emotion: float(prob) 
                            for emotion, prob in zip(EMOTIONS, probs)
                        }
                        results.append({
                            'face_id': i,
                            'emotion': label,
                            'bbox': det['bbox'],
                            'confidence': det['confidence'],
                            'emotion_probabilities': emotion_probs
                        })
                    else:
                        label = emotions_results[i][0]
                        results.append({
                            'face_id': i,
                            'emotion': label,
                            'bbox': det['bbox'],
                            'confidence': det['confidence']
                        })
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return {
        'success': True,
        'faces_count': len(results),
        'results': results,
        'processing_time': processing_time,
        'message': 'No faces detected' if len(results) == 0 else None
    }

# Startup & Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global detector, device
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Initialize face detector
        detector = FaceDetector(
            model_path=Config.YOLO_MODEL_PATH,
            device=str(device),
            conf_threshold=Config.CONF_THRESHOLD,
            iou_threshold=Config.IOU_THRESHOLD
        )
        
        # Initialize emotion classifier
        init_emotion_classifier(device)
        logger.info("Emotion classifier initialized")
        
        logger.info("=" * 60)
        logger.info("Mind Mate API started successfully!")
        logger.info(f"Device: {device}")
        logger.info(f"Docs available at: http://localhost:8000/docs")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Mind Mate API...")

# REST API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mind Mate Emotion Detection API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    is_healthy = detector is not None
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "device": str(device) if device else "unknown",
        "model_loaded": is_healthy,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_emotions_endpoint(file: UploadFile = File(...)):
    """
    Detect emotions from uploaded image
    
    - **file**: Image file (PNG, JPG, JPEG, etc.)
    
    Returns:
    - List of detected faces with emotions and bounding boxes
    """
    try:
        # Read image
        contents = await file.read()
        
        # Check file size
        if len(contents) > Config.MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="Image too large (max 10MB)")
        
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process image
        result = process_image(image, return_probs=False)
        
        logger.info(f"Processed image: {file.filename}, faces: {result['faces_count']}")
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_emotions_endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import Form

@app.post("/predict_simple")
async def predict_simple(file: UploadFile = File(...)):
    try:
        user_id = 1
        department_id = 2

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = process_image(image, return_probs=False)

        if result["faces_count"] == 0:
            emotion = "no_face"
        else:
            emotion = result["results"][0]["emotion"]

        return {
            "user_id": user_id,
            "department_id": department_id,
            "emotion": emotion
        }

    except:
        return {"error": "error"}



'''
#for toekn
@app.post("/predict_simple")
async def predict_simple(
    file: UploadFile = File(...),
    authorization: str = Header(...)
):
    try:
        # 👇 1. فك التوكن
        token = authorization.split(" ")[1]

        
        decoded = jwt.decode(token, "SECRET_KEY", algorithms=["HS256"])
      
        user_id = decoded["user_id"]
        department_id = decoded["department_id"]

        # 👇 2. اقرأ الصورة
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Invalid image"}

        # 👇 3. شغل المودل
        result = process_image(image, return_probs=False)

        if result["faces_count"] == 0:
            emotion = "no_face"
        else:
            emotion = result["results"][0]["emotion"]

        # 👇 4. رجع النتيجة
        return {
            "user_id": user_id,
            "department_id": department_id,
            "emotion": emotion
        }

    except Exception as e:
        return {"error": str(e)}
    
    '''
@app.post("/detect_base64", response_model=DetectionResponse)
async def detect_emotions_base64(request: Base64ImageRequest):
    """
    Detect emotions from base64 encoded image
    
    - **image_base64**: Base64 encoded image string
    
    Returns:
    - List of detected faces with emotions and bounding boxes
    """
    try:
        # Decode image
        image = decode_base64_image(request.image_base64)
        
        # Process image
        result = process_image(image, return_probs=False)
        
        logger.info(f"Processed base64 image, faces: {result['faces_count']}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in detect_emotions_base64: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_with_probabilities", response_model=DetectionResponseWithProbs)
async def detect_with_probabilities_endpoint(file: UploadFile = File(...)):
    """
    Detect emotions with probability scores for each emotion
    
    - **file**: Image file (PNG, JPG, JPEG, etc.)
    
    Returns:
    - List of detected faces with emotions, bounding boxes, and probability scores
    """
    try:
        # Read image
        contents = await file.read()
        
        # Check file size
        if len(contents) > Config.MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="Image too large (max 10MB)")
        
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process image with probabilities
        result = process_image(image, return_probs=True)
        
        logger.info(f"Processed image with probs: {file.filename}, faces: {result['faces_count']}")
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_with_probabilities_endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket Endpoint for Real-time Detection

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time emotion detection
    
    Client sends: Base64 encoded image frames
    Server returns: JSON with detection results
    
    Message format from client:
    {
        "type": "frame",
        "data": "base64_image_string"
    }
    
    Message format to client:
    {
        "type": "result",
        "faces_count": 2,
        "results": [...],
        "processing_time": 0.123
    }
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data.get('type') == 'frame':
                try:
                    # Decode image
                    image = decode_base64_image(data['data'])
                    
                    # Process image
                    result = process_image(image, return_probs=False)
                    
                    # Send result back to client
                    await websocket.send_json({
                        'type': 'result',
                        'success': True,
                        'faces_count': result['faces_count'],
                        'results': result['results'],
                        'processing_time': result['processing_time']
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing frame: {str(e)}")
                    await websocket.send_json({
                        'type': 'error',
                        'success': False,
                        'error': str(e)
                    })
            
            elif data.get('type') == 'ping':
                # Heartbeat
                await websocket.send_json({'type': 'pong'})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

# Additional Utility Endpoints

@app.get("/emotions")
async def get_emotions_list():
    """Get list of supported emotions"""
    return {
        "emotions": list(EMOTIONS),
        "count": len(EMOTIONS)
    }

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    return {
        "active_websocket_connections": len(manager.active_connections),
        "device": str(device),
        "model_loaded": detector is not None
    }

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (للتطوير فقط)
        log_level="info"
    )
