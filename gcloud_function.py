from google.cloud import storage
import cv2
import numpy as np
from scipy.signal import find_peaks

def hello_gcs(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    print(event)
    storage_client = storage.Client()

    bucket_name = 'screw-user-images'
    bucket = storage_client.get_bucket(bucket_name)

    file = event
    blob = bucket.blob(file['name'])

    filestr = blob.download_as_string()
    #convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, 0)

    mm_per_pixel = 0.0714

    fft = FFTObj(img, mm_per_pixel)

    print(f"Processing file: {fft.get_pitch()}.")


class FFTObj:

    def __init__(self, image, pixel_ratio):
        self.img = image
        self.pixel_ratio = pixel_ratio

    def get_fft_mag_spectrum(self):
        (thresh, img2) = cv2.threshold(self.img, 225, 255, cv2.THRESH_BINARY)
        #img2 = cv2.blur(img2, (1, 1280))
        f = np.fft.ifft2(img2)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        return magnitude_spectrum

    def get_fft_graph(self):
        mag = self.get_fft_mag_spectrum()
        graph = list();
        for i in range(len(mag[0])):
            graph.append(0);
        for i in range(len(mag)):
            for j in range(len(mag[i])):
                graph[j] = graph[j] + mag[i][j]
        for i in range(len(graph)):
            graph[i] = graph[i] / len(mag)
        return graph

    def get_peaks(self, graph):
        return find_peaks(graph, threshold=2)[0]

    def get_pitch(self):
        graph = self.get_fft_graph()
        peaks = self.get_peaks(graph)
        maxindex = graph.index(max(graph))
        peakindex = np.where(peaks == maxindex)[0][0]
        pixels = maxindex - peaks[peakindex - 1]
        return pixels * self.pixel_ratio