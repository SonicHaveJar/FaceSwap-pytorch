from imutils.face_utils import FaceAligner
import imutils
import dlib
import cv2
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

def get_faces(video, folder):
    name = os.path.splitext(os.path.basename(video))[0]
    
    if not os.path.exists(f"{folder}/{name}"):
        os.makedirs(f"{folder}/{name}")

    cap = cv2.VideoCapture(video)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        
        # frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:    
            rects = detector(gray, 1)

            faceAligned = fa.align(frame, gray, rects[0])
            cv2.imwrite(f"{folder}/{name}/{name}{count}.jpg", faceAligned)
        except:
            pass

        print(f"Extracting faces: {round(count/frame_count * 100, 2)}% done.")

        count += 1
    
    print(f"DONE!!, faces extracted in {folder}")
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", required=True, help="path to input video")
    parser.add_argument("-f", "--folder", required=True, help="path to data folder")
    args = vars(parser.parse_args())
    
    get_faces(args['video'], args['folder'])
