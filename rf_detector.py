import cv2
import os, os.path
from retinaface import RetinaFace
import numpy as np
import time

'''
Runs the RetinaFace model on frames from a clip.
Also creates writes a file that contains the tuple that is return.
Appends total runtime to a shared file time.txt

@param clip_name: string, the name of the clip as appears in folder/file names
@param generate_video: boolean, true if an annotated video should be generated

@returns a tuple of the single face binary array and the multiple face binary array
'''
def face_detect_rf_seq(clip_name, generate_video):
    print("Starting " + clip_name + " RetinaFace")
    start = time.perf_counter()

    # initialize single/multiple face binary arrays
    path = '../shared/frames/' + clip_name   # path in XSEDE
    num_frames = len([file for file in os.listdir(path)])
    f = [0] * num_frames    # 1 if one face is visible in a frame
    ff = [0] * num_frames   # 1 if multiple faces are visible in a frame

    for i, filename in enumerate(os.listdir(path)):
        frame = os.path.join(path, filename)
        img = cv2.imread(frame)
    
        faces = RetinaFace.detect_faces(img)
        if not (type(faces) is tuple):  # RetinaFace returns a tuple if no face is found
            if len(faces) == 1:
                f[i] = 1
            elif len(faces) > 1:
                ff[i] = 1

            # write annotated image if necessary
            if generate_video:
                # Draw the rectangle around each face
                for key in faces.keys():
                    face = faces[key]
                    rect = face["facial_area"]
                    cv2.rectangle(img, (rect[2], rect[3]), (rect[0], rect[1]), (255, 0, 0), 2)
                cv2.imwrite('annotated/' + clip_name + '_rf/' + filename.split('.')[0] + ".jpg", img)

    if generate_video:
        # compile annotated video
        os.system("ffmpeg -y -framerate 24 -i annotated/" + clip_name + "_rf/" + clip_name + "_%05d.jpg " + "output/" + clip_name + "_output/" + clip_name + "_rf_annotated.mp4")

    # write results to file
    out = open("output/" + clip_name + "_output/" + clip_name + "_rf_output", "wb")
    np.save(out, (f, ff))
    out.close

    # write total runtime to file
    out = open("output/time.txt", "a")
    out.write(clip_name + " RetinaFace took " + str(time.perf_counter() - start) + "\n")
    out.close()
    return (f, ff)

'''
Runs RetinaFace detection on a single image

@param frame: cv image

@returns the annotated image, including face bounding box and facial features
'''
def face_detect_rf(frame):
    faces = RetinaFace.detect_faces(frame)
    if not (type(faces) is tuple):  # RetinaFace returns a tuple if no faces are found
        for key in faces.keys():
            face = faces[key]
            rect = face["facial_area"]
            features = face["landmarks"]
            cv2.rectangle(frame, (rect[2], rect[3]), (rect[0], rect[1]), (255, 0, 0), 2)
            for feature in features.keys():
                cv2.circle(frame, (int(features[feature][0]), int(features[feature][1])), 2, (0, 255, 0), -1)

    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
