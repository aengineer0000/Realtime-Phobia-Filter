# Realtime-Phobia-Filter

## Branch Summary

This branch contains some intial explorations with models, as well as a working pipeline (pipeline/) (video inputs -> prediction -> video output) using a browser UI (Gradio).


## Current State of Pipeline

### Features
- 2 working models (trypophobia (ResNet18), insect (YoloWorld))
- Sliders for users to modify blur strength, conf. threshhold, skipped frames
- Video input and downloadable video output

### Known Issues
- Temp video files are created in the user's default temp folder (..Users/**"your username"**/AppData/Temp), but they seem to persist indefinitely
- Temp video files can currently not be played in the UI, but can be downloaded
It seems the issue has to do with the path that Gradio attempts to access to display the video, but I have not figured out the fix yet
- The insect detection model is still too slow
- The trypophobia model blurs the whole screen when it detects a trypophobic image
