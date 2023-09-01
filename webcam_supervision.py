import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import argparse
import pandas as pd
import xlsxwriter
def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1366, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def main():

    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8n.pt")

    # Tanıma çerçevesi ve bölgeyi başlatın
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    object_counts = {}  # Nesne sayıları için boş bir sözlük oluşturun
    detected_objects = set()  # Algılanmış nesneleri takip etmek için boş bir küme oluşturun

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]

        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.class_id == 0]  # Sadece person sınıfını tanımla

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
        # Algılanmış nesneleri say
        for _, _, _, class_id, _ in detections:
            class_name = model.model.names[class_id]
            key = f"{class_name}"

            # Eğer bu nesne daha önce algılanmadıysa ve sayılmadıysa, sayıyı 1 olarak başlat
            if key not in detected_objects:
                detected_objects.add(key)
                object_counts[key] = 1
            else:
                object_counts[key] += 1  # Eğer daha önce algılandıysa, sayıyı artır

                # Her sınıf için aynı anda görünen nesne sayısını hesaplayın

            class_counts = {}  # Sınıf sayılarını saklamak için bir sözlük oluşturun
            for _, _, _, class_id, _ in detections:
                class_name = model.model.names[class_id]
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1

            # Aynı sınıftan kaç nesne olduğunu ekrana yazdır
            for class_name, count in class_counts.items():
                cv2.putText(frame, f"{class_name}: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        cv2.imshow("Bekir's Cam", frame)
        # Nesne sayısını ekrana kırmızı renkte yazdır


        if cv2.waitKey(30) == 27:
            # Nesne sayılarını bir Excel dosyasına kaydet
            output_filename = "object_counts.xlsx"
            df = pd.DataFrame(list(class_counts.items()), columns=["Object", "Count"])

            with pd.ExcelWriter(output_filename, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="ObjectCounts", index=False)
            break

if __name__ == "__main__":
    main()