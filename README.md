# Digital Out-Of-Home (DOOH) Ad Attention Tracker 👀

Hey everyone! This is my first big computer vision project. It's a tracking script I built to measure how many people are actually looking at digital advertising screens/billboards. It uses realtime face tracking to figure out exactly where someone is looking on a screen and logs it if they stare long enough. 

## Core Features
* **Multi-Face Tracking:** It can track up to 10 people at once! I wrote a custom centroid tracker so the script remembers who is who without mixing them up.
* **3D Math for Gaze Tracking:** Uses OpenCV's `solvePnP` alongside a 3D face model to cast a "ray" from the person's face to figure out the exact X/Y pixels they are looking at.
* **Auto-Calibration:** You don't have to manually type in your screen size. It learns the physical boundaries of your display automatically during the first 6 seconds of running.
* **Blink & Glance Tolerance:** I added a grace period and a smoothing filter. This means it won't cancel a view just because someone blinked or their eyes darted away for a split second.
* **Runs in the Background:** You can set it up to run silently 24/7. I added error handling so it doesn't crash the background process if the webcam suddenly glitches out.

## Configuration
At the top of `core_tracker.py`, I put some variables you can tweak depending on where you set up your screen (like a fast hallway vs. a waiting room):

* `AUTO_CALIB_SECONDS = 6.0`: How long the script watches you at the start to learn the screen's edges. 
* `GAZE_THRESHOLD_SECONDS = 4.0`: The minimum time someone has to look at the screen for it to count as a real "View".
* `GRACE_PERIOD_SECONDS = 1.5`: How long a person can look away (or blink) before their timer completely resets. If they look back before this time is up, the timer keeps going.

## Tech Stack
* Python 3.10
* OpenCV (cv2)
* Google MediaPipe
* NumPy

## Installation
1. Make sure you have Python 3.10 installed.
2. Clone this repo and set up a virtual environment.
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt