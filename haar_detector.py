import cv2
import os, os.path
import numpy as np
import time

'''
Runs the Haar Cascade Classifier on frames from a clip.
Also creates writes a file that contains the tuple that is return.
Appends total runtime to a shared file time.txt

@param clip_name: string, the name of the clip as appears in folder/file names
@param generate_video: boolean, true if an annotated video should be generated

@returns a tuple of the single face binary array and the multiple face binary array
'''
def face_detect_haar_seq(clip_name, generate_video):
    print("Starting " + clip_name + " Haar Cascade")
    start = time.perf_counter()

    # initialize single/multiple face binary arrays
    path = '../shared/frames/' + clip_name
    num_frames = len([file for file in os.listdir(path)])
    f = [0] * num_frames    # 1 if one face is visible in a frame
    ff = [0] * num_frames   # 1 if multiple faces are visible in a frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # the off-the-shelf profile model so far only correctly predicts faces that are already predicted 
    # by the frontal face model, so it's unused here
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    
    for i, filename in enumerate(os.listdir(path)):
        frame = os.path.join(path, filename)
        img = cv2.imread(frame)
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        if len(faces) == 1:
            f[i] = 1
        elif len(faces) > 1:
            ff[i] = 1

        # write annotated image if necessary
        if generate_video:
            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
            cv2.imwrite('annotated/' + clip_name + '_hc/' + filename.split('.')[0] + ".jpg", img)
    
    # compile annotated video
    if generate_video:
        os.system("ffmpeg -y -framerate 24 -i annotated/" + clip_name + "_hc/" + clip_name + "_%05d.jpg " + "output/" + clip_name + "_output/" + clip_name + "_hc_annotated.mp4")

    # write results to file
    out = open("output/" + clip_name + "_output/" + clip_name + "_hc_output", "wb")
    np.save(out, (f, ff))
    out.close

    # write total runtime to file
    out = open("output/time.txt", "a")
    out.write(clip_name + " Haar Cascade took " + str(time.perf_counter() - start) + "\n")
    out.close()

    return (f, ff)

'''
Runs Haar Cascade Classifier for front face and profile face on a single image

@param frame: cv image

@returns the annotated image
'''
def face_detect_haar(frame):
    face_cascade = cv2.CascadeClassifier('C:/Users/avalo/anaconda3/pkgs/libopencv-4.0.1-hbb9e17c_0/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier('C:/Users/avalo/anaconda3/pkgs/libopencv-4.0.1-hbb9e17c_0/Library/etc/haarcascades/haarcascade_profileface.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray)
    # Draw blue rectangle front-facing faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    faces = profile_cascade.detectMultiScale(gray)
    # Draw green rectangle around faces in profile
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)