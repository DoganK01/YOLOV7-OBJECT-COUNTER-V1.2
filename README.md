# Yolov7-Close-People-CounterV1.2

# Track Counter:




https://user-images.githubusercontent.com/98788987/203656286-f8efa500-85cf-483a-aa6c-fa6a174e24ee.mp4






Social Distance with Yolov7



- Count all objetcs by classes and works perfetcly on every image or on a video
- Code can run on Both (CPU & GPU)
- Video/WebCam/External Camera/IP Stream Supported

# Sample Videos:
https://www.youtube.com/watch?v=nB8L54-ejgQ


https://www.youtube.com/watch?v=vANr6aC7QOY
# Ready-To-Use Google Colab:
https://colab.research.google.com/drive/1URG1BruRkVQawZGqCJ1ibV_SeYhEBdKd?usp=sharing
# How to run Code:
- clone the repository:
- `!git clone https://github.com/DoganK01/YOLOV7-OBJECT-COUNTER-V1.2.git`
- `cd /content/YOLOV7-OBJECT-COUNTER-V1.2`

# Upgrade pip with the mentioned command below.
- `pip install --upgrade pip`

# Install requirements
- `!pip install -r requirements.txt`

# Weights get
- `%%bash`
- `wget -P /content/YOLOV7-OBJECT-COUNTER-V1.2/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt`
- `wget -P /content/YOLOV7-OBJECT-COUNTER-V1.2/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt`
- `wget -P /content/YOLOV7-OBJECT-COUNTER-V1.2/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt`
- `wget -P /content/YOLOV7-OBJECT-COUNTER-V1.2/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt`
- `wget -P /content/YOLOV7-OBJECT-COUNTER-V1.2/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt`
- `wget -P /content/YOLOV7-OBJECT-COUNTER-V1.2/weights https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt`
- NOTE: You can use any of these. If one doesn't work, try the other.

# Upgrade pyyaml
- `!pip install --upgrade pyyaml==5.3.1`

# Using Codes:
- For Close People with Red Lines:
- `!python detect.py --weights /content/Yolov7-Close-People-Counter/weights/yolov7x.pt --source /content/Yolov7-Close-People-Counter/inference/images/bus.jpg --no-trace`
- For Close People "Counter":
- `!python counter.py --weights /content/Yolov7-Close-People-Counter/weights/yolov7x.pt --source /content/Yolov7-Close-People-Counter/inference/images/bus.jpg --no-trace`
- For Object Counter:
- `!python detect_and_count.py --weights /content/Yolov7-Close-People-Counter/weights/yolov7x.pt --source /content/Yolov7-Close-People-Counter/inference/images/bus.jpg --no-trace`
# Results
![bus (1)](https://user-images.githubusercontent.com/98788987/188057570-263e4886-29ab-4388-9515-df3ec5f1e359.jpg)


https://user-images.githubusercontent.com/98788987/203448934-37113304-40d3-45ac-8f62-70a14863873f.mp4





https://user-images.githubusercontent.com/98788987/203449682-0fe883a2-d7eb-4398-95e5-96be797b0f5f.mp4

# Real-ESRGAN
Saving people who break the social distance by increasing the resolution from real-video: (Real-ESRGAN):

https://user-images.githubusercontent.com/98788987/218627242-256afe17-e221-432d-9f9f-26ee6fd8c2f0.mp4


# References
- https://github.com/WongKinYiu/yolov7
- https://github.com/xinntao/Real-ESRGAN




