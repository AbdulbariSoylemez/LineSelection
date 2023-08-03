import cv2
import numpy as np

def detect_lane_lines(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Görüntüyü gri tonlamaya dönüştür
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Görüntüye Canny kenar tespiti uygula
        edges = cv2.Canny(gray, 100, 150)

        # ROI (Region of Interest) belirleme

        height, width = edges.shape
        roi_vertices = [
            (width // 10, height - height // 10),  # Sol alt köşe
            (width // 2, height // 2),  # Merkez
            (width - width // 10, height - height // 10)  # Sağ alt köşe
        ]

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, np.array([roi_vertices]), 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Hough çizgi tespiti uygulama
        lines = cv2.HoughLinesP(masked_edges,1, np.pi / 180, 50)

        # Şeritleri görüntü üzerine çizme
        if lines is not None:
            for i in lines:
                x1, y1, x2, y2 = i[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

        cv2.imshow("Lane Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


detect_lane_lines("video/line.mp4")
