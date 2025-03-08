# 📌 Computer Vision Project

Welcome to the **ComputerVision** repository! This project focuses on leveraging **OpenCV** and **MediaPipe** for advanced computer vision applications, including hand tracking, pose estimation, and more. 🚀

---

## Before I start
This was possible because of this course: https://www.youtube.com/watch?v=01sAkU_NvOY

---

## 📖 Project Overview

This repository contains various computer vision implementations utilizing:

- **OpenCV**: Image processing and real-time computer vision.
- **MediaPipe**: Pre-trained machine learning models for facial recognition, hand tracking, pose estimation, etc.
- **Python**: The core programming language for implementing models and processing image data.
- **NumPy**: For mathematical operations.
- **Matplotlib**: For visualization.

---

## 🔧 Installation Guide

Follow these steps to set up the project:

### **1️⃣ Clone the Repository**

```bash
git clone git@github.com:realRSB/ComputerVision.git
cd ComputerVision
```

### **2️⃣ Set Up a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate    # For Windows
```

### **3️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Use

### Hand Tracking

Run the hand tracking script to start the computer vision pipeline:

```bash
python hand_tracking/hand_tracking.py
```

This will start real-time **hand tracking** using a webcam.

### Pose Estimation

Run the personal trainer script for pose estimation and exercise tracking:

```bash
python personal_trainer/personal_trainer.py
```

This script provides real-time pose detection and angle measurement capabilities, tracking various body landmarks and counting repetitions of exercises.

---

## ✨ Features

✔️ **Hand Tracking** – Detect and track hand movements in real time.  
✔️ **Pose Estimation** – Detect body poses and calculate joint angles.  
✔️ **Exercise Repetition Counting** – Count exercise repetitions using pose estimation.  
✔️ **Face Detection** – Identify faces in images and video streams.  
✔️ **Gesture Recognition** – Recognize gestures using MediaPipe models.  
✔️ **Object Detection** – Coming soon!

---

## 🛠️ Tech Stack

| Technology     | Purpose                 |
| -------------- | ----------------------- |
| **Python**     | Core language           |
| **OpenCV**     | Image processing        |
| **MediaPipe**  | Pre-trained ML models   |
| **NumPy**      | Mathematical operations |
| **Matplotlib** | Visualization           |

---

## 📌 To-Do List

- Implement object detection
- Add more exercise routines
- Enhance gesture recognition

---

## ⭐ Show Some Love

If you like this project, give it a ⭐ on GitHub!

Happy coding! 🚀
