LIFT HEIGHT ANALYZER
====================

Lift Height Analyzer is a portable Windows application for automatic analysis of capillary-rise experiment videos.

You do NOT need to install:
- Python
- pip
- OpenCV
- NumPy
- Matplotlib
- VS Code
- any other development software

Internet access is not required to run the application.


QUICK START
===========

1. Extract the downloaded ZIP file.

2. Open the extracted folder:

   LiftHeightAnalyzer

3. Put your experiment folders inside:

   Experiments

   Example:

   LiftHeightAnalyzer
   |
   +-- LiftHeightAnalyzer.exe
   +-- _internal
   +-- README_USER.txt
   |
   +-- Experiments
       |
       +-- 0deg
       |   +-- test_01.mp4
       |   +-- test_02.mp4
       |
       +-- 30deg
       |   +-- test_01.mp4
       |   +-- test_02.mp4
       |
       +-- 60deg
           +-- test_01.mp4
           +-- test_02.mp4

   Each experiment folder may contain one or more MP4 files.

4. Double-click:

   LiftHeightAnalyzer.exe

5. The application automatically uses the included:

   Experiments

   folder.

6. Click:

   Start analysis

7. Wait until the analysis is finished.

8. Results will be created automatically inside:

   Experiments\_analysis


RESULTS
=======

For every analysed video, the program creates:

- a marked MP4 video showing the detected/tracked liquid front;
- a CSV file containing the numerical results;
- a geometry-check PNG;
- a plot showing raw, smoothed and tracked capillary-rise data.

For every experiment folder, the program creates:

- a comparison plot of all tracked runs;
- an average tracked capillary-rise plot;
- an average tracked capillary-rise CSV.

For all experiment folders together, the program creates:

- all_folder_averages.png


IMPORTANT
=========

Keep the complete LiftHeightAnalyzer folder together.

In particular, do NOT move:

   LiftHeightAnalyzer.exe

away from:

   _internal

The _internal folder contains the components required by the application.


OPTIONAL: USING A DIFFERENT DATA FOLDER
=======================================

Normally, you only need to put your experiment folders inside the included:

   Experiments

folder and click Start analysis.

If your measurements are already stored somewhere else, you may instead click:

   Select folder...

and select the folder that directly contains your experiment subfolders.

For example:

   D:\Measurements
   |
   +-- 0deg
   +-- 30deg
   +-- 60deg

The program will then create:

   D:\Measurements\_analysis

This is optional. For normal use, simply use the included Experiments folder.


TRACKING METHOD
===============

The primary result is the tracked capillary-rise curve.

A new higher liquid-front position is accepted only when:

- it is above the previously confirmed position;
- the detections meet the configured detection-quality threshold;
- at least 3 of 5 consecutive frames support approximately the same height.

After a level is confirmed, the tracked physical front is not allowed to decrease.

Experiment-folder averages are calculated from the tracked curves.


WINDOWS SECURITY MESSAGE
========================

Because this is a portable research application and may not be digitally signed, Windows may display a SmartScreen warning the first time it is opened on a new computer.

Only run the application if you received the ZIP from a trusted source.
