LIFT HEIGHT ANALYZER
====================

This is a portable Windows application.

You do NOT need to install:
- Python
- pip
- OpenCV
- NumPy
- Matplotlib
- VS Code
- any development environment

Internet access is not required to run an already-built copy.


HOW TO USE
----------

1. Extract the ZIP.

2. Keep the complete LiftHeightAnalyzer folder together.

3. Double-click:

   LiftHeightAnalyzer.exe

4. Click:

   Select folder...

5. Select the folder containing your experiment subfolders.

Example:

Measurements
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

6. Click:

   Start analysis

7. Results are created in:

   Measurements\_analysis


OUTPUTS
-------

For every video:
- marked MP4
- CSV data
- geometry-check PNG
- individual height plot

For every experiment folder:
- tracked comparison plot
- tracked average plot
- tracked average CSV

For the complete selected experiment root:
- all_folder_averages.png


TRACKING METHOD
---------------

The primary measurement is the tracked capillary-rise curve.

A new higher front is accepted only when:
- it is above the previously confirmed height;
- detections meet the configured quality threshold;
- at least 3 of 5 consecutive frames support roughly the same
  height, within the configured tolerance.

After confirmation, the tracked physical front is not allowed to
decrease.

Folder averages are calculated from the tracked curves.


IMPORTANT
---------

Do not move LiftHeightAnalyzer.exe out of its application folder.

The executable depends on the neighbouring _internal folder bundled
with it.

An unsigned research application may cause Windows SmartScreen to ask
for confirmation the first time it is opened on another computer.
