import cv2
import os, os.path
import mtcnn
import numpy as np
import time

'''
Runs the MTCNN model on frames from a clip.
Also creates writes a file that contains the tuple that is return.
Appends total runtime to a shared file time.txt

@param clip_name: string, the name of the clip as appears in folder/file names
@param generate_video: boolean, true if an annotated video should be generated
@param detector: the mtcnn object -- should be initialized outside of the function for multiple uses

@returns a tuple of the single face binary array and the multiple face binary array
'''
def face_detect_mtcnn_seq(clip_name, generate_video, detector):
    print("Starting " + clip_name + " MTCNN")
    start = time.perf_counter()
    
    # initialize single/multiple face binary arrays
    path = '../shared/frames/' + clip_name   # path in XSEDE
    num_frames = len([file for file in os.listdir(path)])
    f = [0] * num_frames    # 1 if one face is visible in a frame
    ff = [0] * num_frames   # 1 if multiple faces are visible in a frame

    for i, filename in enumerate(os.listdir(path)):
        frame = os.path.join(path, filename)
        img = cv2.imread(frame)
        faces = detector.detect_faces(img)    
        if len(faces) == 1:
            f[i] = 1
        elif len(faces) > 1:
            ff[i] = 1

        # write annotated image if necessary
        if generate_video:
            # Draw the rectangle around each face
            for face in faces:
                rect = face["box"]
                cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)
            cv2.imwrite('annotated/' + clip_name + '_mtcnn/' + filename.split('.')[0] + ".jpg", img)
    
    # compile annotated video
    if generate_video:
        os.system("ffmpeg -y -framerate 24 -i annotated/" + clip_name + "_mtcnn/" + clip_name + "_%05d.jpg " + "output/" + clip_name + "_output/" + clip_name + "_mtcnn_annotated.mp4")

    # write results to file
    out = open("output/" + clip_name + "_output/" + clip_name + "_mtcnn_output", "wb")
    np.save(out, (f, ff))
    out.close
    
    # write total runtime to file
    out = open("output/time.txt", "a")
    out.write(clip_name + " MTCNN took " + str(time.perf_counter() - start) + "\n")
    out.close

    return (f, ff)

'''
Runs MTCNN detection on a single image

@param frame: cv image

@returns the annotated image, including face bounding box and facial features
'''
def face_detect_mtcnn(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detector = mtcnn.MTCNN()
    faces = detector.detect_faces(frame)
    for face in faces:
        rect = face["box"]
        features = face["keypoints"]
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)
        for feature in features.keys():
            cv2.circle(frame, features[feature], 2, (0, 255, 0), -1)

    return frame