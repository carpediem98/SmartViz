# Manufacture Instrument Recognition: SmartViz's Google Cloud Function Backend
This repo contains the testing files for the individual algorithms employed in the backend, as well as ```gcloud_function.py``` to track the exact code in the Cloud Function.
 
## Release Notes

### New Features
* Created ```gcloud_function.py```, which reflects code in cloud
* Encapsulated size detection code into an object for easy use in cloud function
* Encapsulated FFT code into an object for easy use in cloud function
* Created bridge between size detection and FFT to extract necessary details
	
### Bug Fixes
* Removed hardcoded length-per-pixel
* Tuned threshold calculation to prevent crashes

### Known Bugs
* Bad error-handling: crash results in no message being sent back to the client
* Objects without threads cause the backend to crash due to lack of peaks in FFT spectrum


## Install/Testing Guide

### Pre-requisits
This backend runs on Python 3. Please note that you cannot test ```gcloud_function.py``` on local machines, since it is only compatible with the cloud function environment. You can, however, test using ```identify.py```, which is a local version that plots the results into a graph.
	
### Dependencies
Python dependencies:
```
google-cloud-storage
opencv-python
numpy
scipy
imutils
firebase_admin
pyfcm
```

### Local Testing
#### Download
Simply clone this repo.

#### Run
Run:
```
python identify.py
```

### Cloud deployment
#### Upload
* Navigate to the Cloud Console for this project (To request access, contact joel.joseprabu@gmail.com)
* Upload the contents of ```gcloud_function.py``` to the cloud function called ```function1```
* Click Deploy

#### Run
The backend is now deployed. To test it with the UI, please download and run the UI from https://github.com/TimQiu20/Manufacture-Instrument-Recognition-UI.
Please keep in mind that the backend is likely already deployed to the console, as the latest release published here would already have been uploaded. If this is the case, you can simply test by testing the UI.
