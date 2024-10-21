

import time

import pyrealsense2 as rs

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import paho.mqtt.publish as publish

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


shortest_distance = []
count1 = 0
import numpy as np
broker_address = "broker.hivemq.com"
topic = "robot"
Y_REFERENCE_CM = 35
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)


    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',
        source=ROOT / 'data/images',
        data=ROOT / 'data/coco128.yaml',
        imgsz=(640, 640),
        conf_thres=0.1,
        iou_thres=0.45,
        max_det=100,
        device='',
        view_img=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
):


    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280 ,720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    print("[INFO] Starting streaming...")
    pipeline.start(config)
    print("[INFO] Camera ready.")

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights=os.path.join(os.getcwd(),"best (8).pt"), device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    #
    webcam= False
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    while True:
        save_img = True
        path = os.getcwd()
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth = frames.get_depth_frame()

        if not depth: continue

        color_image = np.asanyarray(color_frame.get_data())
        im0s = color_image
        img = letterbox(im0s)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        im = np.ascontiguousarray(img)
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3



        for i, det in enumerate(pred):

            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, s, im0 = path, '', im0s

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                detected_objects =[]
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        d1, d2 = int((int(xyxy[0])+int(xyxy[2]))/2), int((int(xyxy[1])+int(xyxy[3]))/2)
                        zDepth = depth.get_distance(int(d1),int(d2))


                        tl = 3
                        depth_frame = frames.get_depth_frame()
                        tf = max(tl - 1, 1)  # font thickness
                        x_center = int((xyxy[0] + xyxy[2]) / 2)
                        y_center = int((xyxy[1] + xyxy[3]) / 2)
                        CAMERA_L = 1280 // 2
                        CAMERA_B = 720//2


                        cv2.circle(im0 , (CAMERA_L, CAMERA_B), 10, (0,0,255), -1)

                        cv2.circle(im0 , (x_center,y_center), 10, (0,0,255), -1)
                        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                        center_coordinates = rs.rs2_deproject_pixel_to_point(intrinsics, [x_center, y_center],
                                                                            zDepth)

                        center_coordinates_cm = [i * 100 for i in center_coordinates]
                        center_coordinates_cm_y = Y_REFERENCE_CM - center_coordinates_cm[1]
                        if center_coordinates_cm[0] > 0:
                            x_position_1 = abs(center_coordinates_cm[0] * 0.14761)
                            x_position_2 = (x_position_1 * center_coordinates_cm[0]) // 10
                            x_position_final = abs(center_coordinates_cm[0] - x_position_2)

                        elif center_coordinates_cm[0]  < 0:

                            x_position_1 = abs(center_coordinates_cm[0] * 0.14761)
                            x_position_2 = (x_position_1 * center_coordinates_cm[0]) // 10
                            x_position_final = -abs(center_coordinates_cm[0] - x_position_2)

                        detected_objects.append((center_coordinates_cm[2], (center_coordinates_cm[0],center_coordinates_cm_y)))

                        #print("Center Point (x, y, z) in cm:", center_coordinates_cm_y)

                        smallest_depth = 0
                        px = 0
                        py = 0
                        pz = 0
                        roll = 0
                        pitch = 0
                        yaw = 0
                        q1 = 0
                        q2 = 0
                        q3 = 0
                        q4 = 0
                        q5 = 0
                        q6 = 0
                        steps_j1 = 0
                        steps_j2 = 0
                        steps_j3 = 0
                        j2_side = 0


                        cv2.putText(im0, str(round((zDepth* 39.3701 ),2))+"in "+str(round((zDepth* 100 ),2))+" cm" , (d1, d2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                detected_objects.sort(key=lambda x: x[0])
                if detected_objects:
                    if len(detected_objects) >= 3:

                        smallest_depth = detected_objects[0], detected_objects[1], detected_objects[2]

                        z_final = detected_objects[2] - 14
                        px = detected_objects[0] * 100
                        py = detected_objects[1] * 100
                        pz = z_final * 100
                        roll = 0
                        pitch = 90
                        yaw = 0

                        q1, q2, q3, q4, q5, q6 = px, py, pz, roll, pitch, yaw
                        print(q1, q2, q3, q4, q5, q6, "angless")

                    if q1<90:
                        total_steps_j1 = 650
                        min_angle_j1 = 90
                        max_angle_j1 = 45
                        steps_j1 = int((q1 - min_angle_j1) / (max_angle_j1 - min_angle_j1) * total_steps_j1)
                        j1_side = 0
                    elif q1>90:
                        total_steps_j1 = 1450
                        min_angle_j1 = 90
                        max_angle_j1 = 180
                        steps_j1 = int((q1 - min_angle_j1) / (max_angle_j1 - min_angle_j1) * total_steps_j1)
                        j1_side = 1
                    if q2>0:
                        total_steps_j2 = 2470
                        min_angle_j2 = 0
                        max_angle_j2 = 80
                        steps_j2 = int((q2 - min_angle_j2) / (max_angle_j2 - min_angle_j2) * total_steps_j2)
                        j2_side = 1
                    if q2<0:
                        total_steps_j2 = 2470
                        min_angle_j2 = 0
                        max_angle_j2 = 80
                        steps_j2 = int((q2 - min_angle_j2) / (max_angle_j2 - min_angle_j2) * total_steps_j2)
                        j2_side = 0

                    total_steps_j3 = 1270
                    min_angle_j3 = 0
                    max_angle_j3 = 70
                    steps_j3 = int((q3 - min_angle_j3) / (max_angle_j3 - min_angle_j3) * total_steps_j3)
                    if q5>=0:
                        servo = q5 + 90
                    if q5<0:
                        servo = q5 - 90
                    servo_actual = servo - 20


                    publish.single(topic, (str(steps_j1) +" "+str(j1_side) +" "+ str( steps_j2) +" "+ str(j2_side)  +" "+ str(steps_j3)+" "+str(servo_actual)), hostname=broker_address)

                    print(f"Smallest depth of detected object: {smallest_depth}mm")

        cv2.imshow(str(p), im0)
        cv2.waitKey(1)


        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')


    t = tuple(x / seen * 1E3 for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640 , 640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)