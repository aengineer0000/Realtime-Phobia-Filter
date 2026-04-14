# Realtime-Phobia-Filter

## Branch Summary

This branch contains some intial explorations with models, as well as a working pipeline (pipeline/) (video inputs -> prediction -> video output) using a browser UI (Gradio).


## Current State of Pipeline

### Features
- 2 working models (trypophobia (ResNet18), insect (YoloWorld))
- Sliders for users to modify blur strength, conf. threshhold, skipped frames
- Video input; video output downloadable and viewable in UI

### Known Issues
- Temp video files are created in the user's default temp folder (..Users/**"your username"**/AppData/Temp), but they seem to persist indefinitely
- Model inference time for both models still quite slow
- The trypophobia model blurs the whole screen when it detects a trypophobic image; testing a grad-cam fix created patchy blurred images (grad-cam fix is not live)
