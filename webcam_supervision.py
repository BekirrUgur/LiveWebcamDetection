import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import argparse
import pandas as pd
import xlsxwriter

# Sayım alanı çerçevesi belirleniyor
ZONE_POLYGON = np.array([
        [0, 0],
        [0.5, 0],
        [0.5, 1],
        [0, 1]
    ])

def parse_arguments() -> argparse.Namespace: # Çözünürlük
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def main():

    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0) # Pc kamerası "0"
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


    model = YOLO("yolov8n.pt")

    # Tanılama çerçevesi
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    # Sayım kutusu
    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.blue(),
        thickness=7,
        text_thickness=7,
        text_scale=2
    )

    object_counts = {}
    detected_objects = set()

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]

        detections = sv.Detections.from_ultralytics(result)

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        zone.trigger(detections=detections)

        # Nesne sayılarını güncelle
        for _, _, _, class_id, _ in detections:
            class_name = model.model.names[class_id]

            if class_name not in detected_objects: # Her nesneyi bir kez saydırır
                detected_objects.add(class_name)  # Nesneyi set'e ekle
                if class_name in object_counts:
                    object_counts[class_name] += 1
                else:
                    object_counts[class_name] = 1

        frame = zone_annotator.annotate(scene=frame)

        cv2.imshow("Bekir's Cam", frame)

        if(cv2.waitKey(30) == 27):
            # Belirlediğim nesneleri excel dosyası formatında kaydediyor
            output_filename = "object_counts.xlsx"
            df = pd.DataFrame(list(object_counts.items()), columns=["Object", "Count"])

            with pd.ExcelWriter(output_filename, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="ObjectCounts", index=False)
            break



if __name__ == "__main__":
    main()