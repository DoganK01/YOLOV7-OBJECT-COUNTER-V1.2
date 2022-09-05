import argparse
from os import O_RDONLY
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from math import sqrt
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import os

def draw_line(img,left,right,color=(0,0,225),line_thickness=1):

  cv2.line(img, left, right, color, thickness=line_thickness) 



def counter(founded_people,xyxy,q,top,a):
  founded_people[f"people{q}"] = [int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])]               
  q = q + 1
  top = top + xyxy[3] - xyxy[1]
  a = True 

  return founded_people,q,top,a

def distance_counter(img,founded_people,top,q,h,plot_box,det,save_img,b,draw):
  #m = people0 #k = [10, 1.0, 12, 7.5] 
  for ş, (m, k) in enumerate(founded_people.items()):
    for o, (j, l) in enumerate(founded_people.items()):
      if not m == j:
        left_1 = np.array([k[0], k[3]])
        right_1 = np.array([k[2], k[3]])

        left_2 = np.array([l[0], l[3]])
        right_2 = np.array([l[2], l[3]])
        
        check_point1 = right_1 - left_2
        check_point2 = left_1 - right_2
        checks1 = sqrt((check_point1[0]**2) + (check_point1[1]**2))  
        checks2 = sqrt((check_point2[0]**2) + (check_point2[1]**2))

        if (checks1 < int((top)/q)/1.2573): #75
          h += 1
          plot_box (img,m,j,founded_people)
          if save_img:
            b = save(img,l,b)
          if draw:
            draw_line(img,(k[2],k[3]),(l[0],l[3]))


        if (checks2 < int((top)/q)/1.2573):
          h += 1
          plot_box (img,m,j,founded_people)
          if save_img:
            b = save(img,l,b)
          if draw:
            draw_line(img,(k[0],k[3]),(l[2],l[3]))
                    
  # cv2.putText(img,f"Close People in Pairs: = {int(h/2)} ",(0, 105), cv2.FONT_HERSHEY_TRIPLEX,1, (255, 0, 0), 1)
  cv2.putText(img,f"Total Object: = {len(det)}",(0, 255), cv2.FONT_HERSHEY_TRIPLEX,1, (200, 100, 0), 1)



def plot_box (image,m,j,founded_people,color=(0,0,225),line_thickness=3):
  tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
  c1, c2= (int(founded_people[m][0]), int(founded_people[m][3])),(int(founded_people[m][2]), int(founded_people[m][3]))
  k1, k2= (int(founded_people[j][0]), int(founded_people[j][3])),(int(founded_people[j][2]), int(founded_people[j][3]))
  cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
  cv2.rectangle(image, k1, k2, color, thickness=tl, lineType=cv2.LINE_AA)

def save(im0,l,b):
  bb=im0[int(l[1]):int(l[3]),int(l[0]):int(l[2])]
  path="/content/gdrive/MyDrive/yolov7/people_imgs"
  bb_resized=cv2.resize(bb,(224,224))
  
  cv2.imwrite(os.path.join(path,f"people_close{b}.jpg"),bb_resized)
  b = b + 1
  return b

"""
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
"""

def count(classes,image):
  model_values=[]
  aligns=image.shape
  align_bottom=aligns[0]
  align_right=(aligns[1]/1.7 ) 

  for i, (k, v) in enumerate(classes.items()):
    a=f"{k} = {v}"
    model_values.append(v)
    align_bottom=align_bottom-35                                                   
    cv2.putText(image, str(a) ,(int(align_right),align_bottom), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),1,cv2.LINE_AA)



def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, save_img, draw = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.save_img, opt.draw
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    b = 0
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                founded_classes={}
                founded_people={}
                q = 0
                h = 0
                a = None
                top = 0
                


                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    class_index=int(c)
                    count_of_object=int(n)
                    founded_classes[names[class_index]]=int(n)
                    count(classes=founded_classes,image=im0)
         
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    
                    if save_txt: # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                          f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if cls == 0:
                      founded_people,q,top,a = counter(founded_people,xyxy,q,top,a)
                      """
                      left_middle = [int(xyxy[0]), int(xyxy[3])]
                      right_middle = [int(xyxy[2]),int(xyxy[3])]
                  
                      """
                                         
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        
                    
                        
                if a == True:
                  distance_counter(im0,founded_people,top,q,h,plot_box,det,save_img,b,draw)
                         
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--save_img', default='store_true', help='Save close people images')
    parser.add_argument('--draw', default='store_true', help='Draw Line between close People')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
