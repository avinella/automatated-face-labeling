import cv2
import os, os.path
import mtcnn
from retinaface import RetinaFace
import numpy as np
import time

from mtcnn_detector import face_detect_mtcnn_seq
from haar_detector import face_detect_haar_seq
from rf_detector import face_detect_rf_seq

'''
Scrapes a clip-specific handcoding to create the ground truth binary arrays
for single and multiple faces

@param scence_labels: string path to the handcoding txt file
@returns a tuple of the single and multiple face binary arrays
'''
def scrape_labels(scene_labels):
    # initialize number of frames
    num_frames = 0
    with open(scene_labels, "r") as file:
        for line in file:
            # all codings list the final frame
            if line[:3] == "end":
                split = line.split()
                num_frames = split[1]
                break

    print("num frames: " + num_frames)
        
    f = [0] * int(num_frames)    # 1 if one face is visible in a frame
    ff = [0] * int(num_frames)   # 1 if multiple faces are visible in a frame

    # find which frames should have the appropriate binary label
    with open(scene_labels, "r") as file:
        for line in file:
            if line[:2] != "//" and line[0] == 'f':
                split = line.split()
                start = int(split[1])
                end = int(split[2])
                if line[1] == 'f':
                    for i in range(start-1, end):
                        ff[i] = 1
                else:
                    for i in range(start-1, min(end, int(num_frames))): 
                        f[i] = 1
                    
    return (f, ff)

'''
Runs each detector on a single clip. Note that for highest efficiency, it would be
better to run each model in parallel if possible.

@param clipname: string, name of clip
@param generate_video: boolean, true if an annotated video should be generated
'''
def run_detectors(clipname, generate_video):
    # make necessary folders for output if needed
    if generate_video:
        os.system("mkdir annotated/" + clipname + "_hc")
        os.system("mkdir annotated/" + clipname + "_rf")
        os.system("mkdir annotated/" + clipname + "_mtcnn")
    os.system("mkdir output/" + clipname + "_output")
    
    face_detect_rf_seq(clipname, generate_video)
    face_detect_haar_seq(clipname, generate_video)
    
    detector = mtcnn.MTCNN()    # initialize detector first -- takes a long time
    face_detect_mtcnn_seq(clipname, generate_video, detector)


'''
Compare the single/multiple face binary arrays from running a model on a clip to 
the same ground truth arrays.

@param results_f: single face binary array from model
@param results_ff: multiple face binary array from model
@param ground_f: single face binary array from handcoding
@param ground_ff: multiple face binary array from handcoding

@returns a tuple with the following information:
    Total number of frames with incorrect labels
    % frames with the correct single face label
    % frames with the correct multiple face label
    List of frame numbers with incorrect single face label
    List of frame numbers with incorrect multiple face label
'''
def compare_face_detect(results_f, results_ff, ground_f, ground_ff):
    # must have the same number of frames
    if (not len(results_f) == len(ground_f)) or (not len(results_ff) == len(ground_ff)):
        return (-1, 0, 0, [], [])

    wrong_f = []      # list of frames with incorrect labels in face
    for i, face in enumerate(results_f):
        if not face == ground_f[i]:
            wrong_f.append(i)

    wrong_ff = []      # list of frames with incorrect labels in faces
    for i, faces in enumerate(results_ff):
        if not faces == ground_ff[i]:
            wrong_ff.append(i)

    # For total incorrect frames, only count overlapping frames once
    missed = len(wrong_f) + len(wrong_ff) - len([x for x in wrong_ff if x in wrong_f])

    return (missed, 1 - (len(wrong_f)/len(results_f)), 1 - (len(wrong_ff)/len(results_ff)), wrong_f, wrong_ff)


'''
Writes model and ground truth comparison results to a file

@param model: string, the name of the model (MUST be "hc", "mtcnn", or "rf")
'''
def generate_results(model):
    # goes through all clip results
    path = '../shared/frames/'  # path in XSEDE
    for clipname in os.listdir(path):
        # gets handcoded lables
        ground = scrape_labels('hand_coding/' + clipname + '_hcode.txt')

        # file that contains model output data
        file = open('output/' + clipname + '_output/' + clipname + '_' + model + '_output', "rb")
        result = np.load(file)

        missed, acc_f, acc_ff, wrong_f, wrong_ff = compare_face_detect(result[0], result[1], ground[0], ground[1])

        # writes results to a file named [modelname]_results.txt
        results = open('output/' + model + '_results.txt', "a")
        results.write("\nClip: " + clipname + '\n')
        results.write(model + "missed: " + str(missed) + " out of " + str(len(result[0])) + "\nSingle face accuracy: " + str(acc_f) + "\nMulti face accuracy: " + str(acc_ff) +
            "\nTotal accuracy: " + str(missed/len(result[0])) + "\n")

        results.close()
        file.close()

'''
Finds the names of all the clips based on the folders in the "frames" folder

@returns an array of the clip names as strings
'''
def get_clipnames():
    clipnames = []
    path = '../shared/frames/'  # path in XSEDE
    for clipname in os.listdir(path):
        clipnames.append(clipname)
    return clipnames

'''
Runs each detector on a set of clips

@param clips: list of clip names. If the list is empty, it will process all clips in the "frames" folder
@param generate_video: boolean, true if an annotated video should be generated
'''
def main(clips, generate_video):
    if not clips:
        path = '../shared/frames/'  # path in XSEDE
        for clipname in os.listdir(path):
            run_detectors(clipname, generate_video)
    else:
        for clipname in clips:
            run_detectors(clipname, generate_video)

# runs all models on all clips, then generates the overall results
main([], False)
models = ["hc", "mtcnn", "rf"]
for model in models:
    generate_results(models)
    
