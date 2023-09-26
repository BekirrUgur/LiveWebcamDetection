import time
import pandas as pd
import cv2
import argparse
import json
from datetime import datetime
import sys

import numpy as np
from ultralytics import YOLO
import supervision as sv
#
#http://194.44.38.196:8083/ Ukraine Lviv
#http://158.58.130.148:80/mjpg/video.mjpg rusya hotel
"""START = sv.Point(425, 200)
END = sv.Point(125, 325)"""
#http://128.101.85.194:80/mjpg/video.mjpg minneapolis USA
"""START = sv.Point(580, 480)
END = sv.Point(400, 500)"""

http = "http://128.101.85.194:80/mjpg/video.mjpg"
START = sv.Point(580, 480)
END = sv.Point(400, 500)
_line_zone_in = 0
_line_zone_out = 0
def dataToExcel(in_count, out_count):
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        # Mevcut veriyi yükle
        df = pd.read_excel('veriler.xlsx')
    except FileNotFoundError:
        # Eğer dosya bulunamazsa, yeni bir DataFrame oluştur
        df = pd.DataFrame(columns=['Time', 'in', 'Out'])

    # Yeni veriyi oluştur
    yeni_veri = {'Time': time,
                 'in': in_count,
                 'Out': out_count}

    # Yeni veriyi DataFrame'e ekleyerek birleştir
    df = pd.concat([df, pd.DataFrame([yeni_veri])], ignore_index=True)

    # Güncellenmiş veriyi kaydet
    df.to_excel('veriler.xlsx', index=False)

def main():
    model=YOLO("yolov8n.pt")
    line_zone=sv.LineZone(start=START, end=END)
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=0.5
    )

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    for result in model.track(source=http, show=True, stream=True):
        frame = result.orig_img
        detections = sv.Detections.from_ultralytics(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        detections = detections[detections.class_id == 0]

        labels = [
            f"#{tracker_id}{model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )
        line_zone.trigger(detections=detections)
        line_zone_annotator.annotate(frame=frame, line_counter=line_zone)

        print("Giren",line_zone.in_count)
        print("Çıkan", line_zone.out_count)
        #dataToExcel(line_zone.in_count,line_zone.out_count)
        global _line_zone_in
        global _line_zone_out
        if _line_zone_in < line_zone.in_count or _line_zone_out < line_zone.out_count:
            _line_zone_in = line_zone.in_count
            _line_zone_out = line_zone.out_count
            dataToExcel(_line_zone_in, _line_zone_out)

        cv2.imshow("yolov8",frame)

        if (cv2.waitKey(1) == 27):
            break

if __name__ == "__main__":
    main()
