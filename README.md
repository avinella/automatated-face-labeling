# Comparing Deep Learning Models for Automated Facial Labeling
CS153 Final Project

This project compares the accuracy of three deep learning face detection models, a Haar cascade classifier, a Multi-tasking cascade convolutional network (MTCNN), and RetinaFace, on short film clips. These were run on the XSEDE bridges2 supercomputer on 2 GPUs. For both space and copyright purposes, the original clips as well as their separated frames are not included in this repository, but select example annotated clips and frames are included for reference. In addition, the output files for each model on every clip and their final results are included.

The documentation for the specific implementation of each model can be found at these links:
- [Haar cascade classifier](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
- [MTCNN](https://pypi.org/project/mtcnn/)
- [RetinaFace](https://github.com/serengil/retinaface)

The frames from the clips were generated using [ffmpeg](https://ffmpeg.org/) with:

`ffmpeg -i clipname.mp4 '/frames/clipname/clipname_%05d.png`

and restitched together with

`ffmpeg -y -framerate 24 -i annotated/" + clip_name + "_hc/" + clip_name + "_%05d.jpg " + "output/" + clip_name + "_output/" + clip_name + "_hc_annotated.mp4`.

Note that the annotated frames should be converted to JPEGs before stitching, as the PNG files may be too large to properly view in an MP4.

The results of each model were compared to hand-labeled data provided by [Gaze Data for the Analysis of Attention in Feature Films](https://graphics.stanford.edu/~kbreeden/gazedata.html).

main.py can be run in its current state to run every model on every clip without generating annotated frames; however, the clip frames are required to run. They are expected to be in the path `../shared/frames/`.
