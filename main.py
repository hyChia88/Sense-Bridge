import cv2
import mediapipe as mp
import pygame
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import tkinter.messagebox as messagebox
from tkinter import filedialog

class SimpleModelViewer:
    def __init__(self, canvas):
        self.canvas = canvas
        self.angle = 0
        
        # Configure canvas
        self.canvas.config(width=300, height=300, bg='white')
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Draw initial shape
        self.draw()
        
    def on_click(self, event):
        self.lastX = event.x
        
    def on_drag(self, event):
        deltaX = event.x - self.lastX
        self.angle += deltaX * 0.5
        self.lastX = event.x
        self.draw()
        
    def draw(self):
        self.canvas.delete("all")
        
        # Center of canvas
        cx, cy = 150, 150
        size = 100
        
        # Calculate corners of a square based on rotation angle
        rad = np.radians(self.angle)
        points = []
        for i in range(4):
            angle = rad + np.pi * i / 2
            x = cx + size * np.cos(angle)
            y = cy + size * np.sin(angle)
            points.extend([x, y])
            
        # Draw rotated square
        self.canvas.create_polygon(points, fill='lightblue', outline='blue')
        
        # Draw some details
        self.canvas.create_line(cx, cy, cx + size * np.cos(rad), cy + size * np.sin(rad), 
                              fill='red', width=2)
        self.canvas.create_oval(cx-5, cy-5, cx+5, cy+5, fill='red')

class VideoCallEnhancer:
    def __init__(self, window):
        self.window = window
        self.window.title("Video Call Enhancer")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot access webcam!")
            self.window.destroy()
            return
        
        # Initialize MediaPipe components
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mpFace = mp.solutions.face_detection
        self.faceDetection = self.mpFace.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=1
        )
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mpDraw = mp.solutions.drawing_utils
        self.fistDetected = False
        
        # Initialize audio
        pygame.mixer.init()
        self.music = pygame.mixer.music
        try:
            self.music.load('background_music.mp3')
            self.defaultVolume = 1.0
            self.lowVolume = 0.1
            self.currentVolume = self.defaultVolume
            self.music.set_volume(self.defaultVolume)
        except:
            messagebox.showwarning("Warning", "Background music file not found!")
        
        # Parameters
        self.frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.distanceThresholdHi = 0.3
        self.distanceThresholdLo = 0.1
        self.maxZoom = 2.0
        self.minZoom = 1.0
        self.zoomSpeed = 0.05
        self.last_talking_time = 0
        self.zoom_cooldown = 2.0
        
        # Mouth landmarks indices
        self.upperLipIndices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 95, 88, 178, 87]
        self.lowerLipIndices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
        
        # States
        self.mouthOpenThreshold = 0.015
        self.mouthOpenHistory = []
        self.mouthHistorySize = 5
        self.isRunning = False
        self.isTracking = False
        self.isManualZoom = False
        self.currentZoom = 1.0
        self.zoomingIn = True
        self.activeFace = None
        
        # Setup GUI
        self.setupGUI()

    def setupGUI(self):
        # Configure main window
        self.window.configure(bg='#2d2d2d')
        
        # Main container
        mainContainer = ttk.Frame(self.window)
        mainContainer.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Left panel for video
        leftPanel = ttk.Frame(mainContainer)
        leftPanel.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        
        # Video display
        self.videoLabel = ttk.Label(leftPanel)
        self.videoLabel.pack(padx=5, pady=5)
        
        # Controls container
        controlsFrame = ttk.Frame(leftPanel)
        controlsFrame.pack(padx=5, pady=5, fill=tk.X)
        
        # Volume display frame
        volumeFrame = ttk.LabelFrame(leftPanel, text="Volume Control")
        volumeFrame.pack(padx=5, pady=5, fill=tk.X)
        
        # Volume progress bar
        self.volumeBar = ttk.Progressbar(volumeFrame, length=200, mode='determinate')
        self.volumeBar.pack(padx=5, pady=5)
        
        # Volume label
        self.volumeLabel = ttk.Label(volumeFrame, text="Volume: 100%")
        self.volumeLabel.pack(padx=5, pady=5)
        
        # Start/Stop button
        self.trackingButton = ttk.Button(
            controlsFrame,
            text="Start Tracking",
            command=self.toggleTracking
        )
        self.trackingButton.pack(side=tk.LEFT, padx=5)
        
        # Close button
        closeButton = ttk.Button(
            controlsFrame,
            text="Close",
            command=self.cleanup
        )
        closeButton.pack(side=tk.RIGHT, padx=5)
        
        # Right panel for simple viewer and controls
        rightPanel = ttk.Frame(mainContainer)
        rightPanel.pack(side=tk.RIGHT, padx=5, fill=tk.BOTH)
        
        # Simple Viewer
        viewerFrame = ttk.LabelFrame(rightPanel, text="Model Viewer")
        viewerFrame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # Create simple canvas
        self.canvas = tk.Canvas(viewerFrame, width=300, height=300)
        self.canvas.pack(padx=5, pady=5)
        
        self.modelViewer = SimpleModelViewer(self.canvas)
        
        # Status frame
        self.statusFrame = ttk.LabelFrame(rightPanel, text="Status")
        self.statusFrame.pack(padx=5, pady=5, fill=tk.X)
        
        # Status labels
        self.faceCountLabel = ttk.Label(self.statusFrame, text="Faces: 0")
        self.faceCountLabel.pack(anchor=tk.W, padx=5, pady=2)
        
        self.talkingLabel = ttk.Label(self.statusFrame, text="Talking: No")
        self.talkingLabel.pack(anchor=tk.W, padx=5, pady=2)
        
        self.zoomLabel = ttk.Label(self.statusFrame, text="Zoom: 1.0x")
        self.zoomLabel.pack(anchor=tk.W, padx=5, pady=2)
        
        self.mouthStateLabel = ttk.Label(self.statusFrame, text="Fist: No")
        self.mouthStateLabel.pack(anchor=tk.W, padx=5, pady=2)

    def updateVolume(self, volume):
        self.currentVolume = volume
        percentage = int(volume * 100)
        self.volumeBar['value'] = percentage
        self.volumeLabel.configure(text=f"Volume: {percentage}%")
        self.music.set_volume(volume)

    def processFrame(self, frame):
        if not self.isTracking:
            return frame, False, []
        
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = []
        isTalking = False
        
        # Process face mesh for mouth detection
        meshResults = self.faceMesh.process(rgbFrame)
        
        if meshResults.multi_face_landmarks:
            for faceLandmarks in meshResults.multi_face_landmarks:
                mouthOpen = self.isMouthOpen(faceLandmarks.landmark)
                self.mouthOpenHistory.append(mouthOpen)

                if len(self.mouthOpenHistory) > self.mouthHistorySize:
                    self.mouthOpenHistory.pop(0)
                
                isTalking = any(self.mouthOpenHistory)
                
                for idx in self.upperLipIndices + self.lowerLipIndices:
                    point = faceLandmarks.landmark[idx]
                    x = int(point.x * self.frameWidth)
                    y = int(point.y * self.frameHeight)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Process face detection
        faceResults = self.faceDetection.process(rgbFrame)
        if faceResults.detections:
            for detection in faceResults.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * self.frameWidth)
                y = int(bbox.ymin * self.frameHeight)
                w = int(bbox.width * self.frameWidth)
                h = int(bbox.height * self.frameHeight)
                faces.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Process hands
        handResults = self.hands.process(rgbFrame)
        self.fistDetected = False
        
        if handResults.multi_hand_landmarks:
            for handLandmarks in handResults.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, handLandmarks, 
                                         self.mpHands.HAND_CONNECTIONS)
                if self.detectFist(handLandmarks):
                    self.fistDetected = True
                    self.updateVolume(self.lowVolume)
                else:
                    self.updateVolume(self.defaultVolume)
        
        frame = self.handleZoom(frame, faces, isTalking, self.fistDetected)
        return frame, isTalking, faces

    # [Rest of the methods remain the same as in your original code]
    def isMouthOpen(self, landmarks):
        if not landmarks:
            return False
        upperLipY = np.mean([landmarks[idx].y for idx in self.upperLipIndices])
        lowerLipY = np.mean([landmarks[idx].y for idx in self.lowerLipIndices])
        mouthOpening = abs(upperLipY - lowerLipY)
        return mouthOpening > self.mouthOpenThreshold

    def detectFist(self, handLandmarks):
        if not handLandmarks:
            return False
        tipIds = [4, 8, 12, 16, 20]
        baseIds = [2, 6, 10, 14, 18]
        isFist = True
        for tip, base in zip(tipIds, baseIds):
            if handLandmarks.landmark[tip].y < handLandmarks.landmark[base].y:
                isFist = False
                break
        return isFist

    def handleZoom(self, frame, faces, isTalking, fistDetected):
        # [Same as original code]
        return frame

    def zoomToFace(self, frame, faceBox):
        # [Same as original code]
        return frame

    def updateFrame(self):
        while self.isRunning:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame, isTalking, faces = self.processFrame(frame)
            
            self.window.after(0, self.updateStatus, len(faces), isTalking, self.currentZoom)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            self.window.after(0, self.updateVideo, photo)

    def updateStatus(self, faceCount, isTalking, zoomFactor):
        self.faceCountLabel.configure(text=f"Faces: {faceCount}")
        self.talkingLabel.configure(text=f"Talking: {'Yes' if isTalking else 'No'}")
        self.zoomLabel.configure(text=f"Zoom: {zoomFactor:.1f}x")
        self.mouthStateLabel.configure(text=f"Fist: {'Yes' if self.fistDetected else 'No'}")

    def updateVideo(self, photo):
        self.videoLabel.configure(image=photo)
        self.videoLabel.image = photo

    def toggleTracking(self):
        self.isTracking = not self.isTracking
        if self.isTracking:
            self.trackingButton.configure(text="Stop Tracking")
            if not self.isRunning:
                self.isRunning = True
                self.music.play(-1)
                threading.Thread(target=self.updateFrame, daemon=True).start()
        else:
            self.trackingButton.configure(text="Start Tracking")
            self.updateVolume(self.defaultVolume)

    def cleanup(self):
        self.isRunning = False
        self.isTracking = False
        if self.cap.isOpened():
            self.cap.release()
        pygame.mixer.quit()
        self.window