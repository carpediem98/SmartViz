import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(sys.argv[1],0)
ogimg = img
(thresh, img) = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY)
#img = cv2.blur(img, (1, 1280))

f = np.fft.ifft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

graph = list();

for i in range(len(magnitude_spectrum[0])):
    graph.append(0);

for i in range(len(magnitude_spectrum)):
    for j in range(len(magnitude_spectrum[i])):
        graph[j] = graph[j] + magnitude_spectrum[i][j]

for i in range(len(graph)):
    graph[i] = graph[i] / len(magnitude_spectrum)


plt.subplot(221),plt.imshow(ogimg, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(222),plt.imshow(img, cmap = 'gray')
plt.title('Processed Image'), plt.xticks([]), plt.yticks([])

plt.subplot(224),plt.plot(graph)
plt.title('FFT'), plt.xticks([]), plt.yticks([])

# plt.subplot(224),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('FFT'), plt.xticks([]), plt.yticks([])

plt.show()