# Lift Height Analyzer

Lift Height Analyzer is a portable Windows application for automatic analysis of capillary-rise experiment videos.

You do **not** need to install Python, pip, OpenCV, NumPy, Matplotlib, VS Code, or any other development software.

Internet access is not required to run the already-built application.

## Quick start

1. Extract the downloaded ZIP file.

2. Open the extracted `LiftHeightAnalyzer` folder.

3. Put your experiment folders inside the included `Experiments` folder.

Example:

```text
LiftHeightAnalyzer/
├── LiftHeightAnalyzer.exe
├── _internal/
├── README_USER.txt
├── README_USER.md
└── Experiments/
    ├── 0deg/
    │   ├── test_01.mp4
    │   └── test_02.mp4
    ├── 30deg/
    │   ├── test_01.mp4
    │   └── test_02.mp4
    └── 60deg/
        ├── test_01.mp4
        └── test_02.mp4
```

Each experiment folder may contain one or more `.mp4` files.

4. Double-click `LiftHeightAnalyzer.exe`.

5. The application automatically uses the included `Experiments` folder.

6. Click **Start analysis**.

7. Wait until the analysis is finished.

8. Results are created automatically in:

```text
Experiments\_analysis
```

## Results

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

- `all_folder_averages.png`

## Important

Keep the complete `LiftHeightAnalyzer` folder together.

Do **not** move `LiftHeightAnalyzer.exe` away from the `_internal` folder. The `_internal` folder contains components required by the application.

## Optional: use a different data folder

Normally, simply put your experiment folders inside the included `Experiments` folder and click **Start analysis**.

If your measurements are already stored somewhere else, click **Select folder...** and select the folder that directly contains the experiment subfolders.

Example:

```text
D:\Measurements\
├── 0deg\
├── 30deg\
└── 60deg\
```

The program then creates:

```text
D:\Measurements\_analysis
```

This is optional. For normal use, use the included `Experiments` folder.

## Tracking method

The primary result is the tracked capillary-rise curve.

A new higher liquid-front position is accepted only when:

- it is above the previously confirmed position;
- the detections meet the configured detection-quality threshold;
- at least 3 of 5 consecutive frames support approximately the same height.

After a level is confirmed, the tracked physical front is not allowed to decrease.

Experiment-folder averages are calculated from the tracked curves.

## Windows security message

Because this is a portable research application and may not be digitally signed, Windows may display a SmartScreen warning the first time it is opened on a new computer.

Only run the application if you received the ZIP from a trusted source.
