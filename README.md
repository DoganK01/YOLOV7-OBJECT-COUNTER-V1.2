# Yolov7-Close-People-CounterV1.2

Counting Close People with Yolov7


- Censoring and counting all cigarettes on the screen and saving smoker's faces in the folder or any database
- Useful for public non-smoking areas
- Code can run on Both (CPU & GPU)
- Video/WebCam/External Camera/IP Stream Supported

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

# Using Counter Example:
- `!python detect.py --weights /content/Yolov7-Close-People-Counter/weights/yolov7x.pt --source /content/Yolov7-Close-People-Counter/inference/images/bus.jpg --no-trace`
# Results
![bus](https://user-images.githubusercontent.com/98788987/187685498-012c769f-3d99-48a7-ba4d-0e01cd71420d.jpg)
# Sample Video:
https://www.youtube.com/watch?v=fnMa_puCUow
