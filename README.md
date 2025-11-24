# **Drone Detection System (YOLOv8m \+ Streamlit App)**

A real-time **drone detection system** built using **YOLOv8m**, fine-tuned on a custom dataset from Roboflow. The system can detect drones in the sky even in the presence of birds and process full videos **frame-by-frame** with high accuracy.  
 A clean **Streamlit web app** is included for easy inference and exporting annotated videos.

---

## **🚀 Project Overview**

Modern security systems rely heavily on automated aerial object monitoring. This project delivers:

* Reliable **drone detection** in varied backgrounds

* Robust performance even when **drones are surrounded by birds**

* Support for **images and full video pipelines**

* A lightweight **Streamlit UI** for local deployment

* Downloadable **fully annotated video output**

---

## 

## 

## 

## **✨ Features**

### **🎯 YOLOv8m-based Drone Detection**

* Fine-tuned YOLOv8m.pt on a curated Roboflow dataset

* Solid performance in challenging lighting and cluttered sky scenes

* Real-time inference (GPU recommended)

### **🎥 Video Processing Pipeline**

* Takes uploaded videos

* Breaks them into frames

* Runs YOLOv8m inference

* Reconstructs a fully annotated output video

* Output is downloadable from the Streamlit UI

### **🖥️ Streamlit App**

* Clean, minimal UI  
* Supports: **image upload**, **video upload**  
* Instant preview \+ downloadable results

---

## **🧪 Training Summary**

* **Base model:** YOLOv8m

* **Framework:** Ultralytics YOLOv8

* **Dataset provider:** Roboflow

* **Dataset class:** drone

* **Training steps:**

  * Model fine-tuning

  * Validation & testing

  * Performance evaluation using Precision, Recall, mAP50  mAP50-95

---

## **📈 Results**

| Metric | Score |
| ----- | :---- |
| mAP50 | 94.2% |
| mAP50-95 | 73% |
| Precision | 93.1% |
| Recall | 91.5% |

---

## 

## **🛠️ Installation**

`git clone https://github.com/your-username/drone-detection`  
`cd drone-detection`  
`pip install -r requirements.txt`

---

## **▶️ Running the Streamlit App**

`streamlit run DRONE_DETECTION_APP.py`

Upload an image/video → wait for processing → download the annotated output.

---

## 

## **📌 Future Improvements**

* ✔ **SORT / DeepSORT tracking pipeline**

* ✔ Drone movement trajectory plotting

* ✔ Threat-level scoring (size, speed, movement pattern)

* ✔ Multi-camera input support

* ✔ Real-time alert system

* ✔ ONNX export and edge-device deployment

---

## **📜 License**

MIT License

---

## **🤝 Contributing**

Pull requests are welcome\!

