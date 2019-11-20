import cv2
import numpy as np
from scipy.signal import find_peaks

class FFTObj:

    def __init__(self, image, pixel_ratio):
        self.img = cv2.imread(image,0)
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
        thresh = (max(graph) - min(graph)) * 0.01
        print(thresh)
        return find_peaks(graph, threshold=thresh)[0]

    def get_pitch(self):
        graph = self.get_fft_graph()
        peaks = self.get_peaks(graph)
        maxindex = graph.index(max(graph))
        peakindex = np.where(peaks == maxindex)[0][0]
        pixels = maxindex - peaks[peakindex - 1]
        return pixels * self.pixel_ratio
