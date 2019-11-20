from matplotlib import pyplot as plt
from fft.FFTObj import FFTObj
from ML.SizeObj import SizeObj
import cv2

image = "withScrewTest.jpg"
# image = "fft/screw.jpg"

size_obj = SizeObj(image)
size_obj.process(0.955)

length = size_obj.sizes[1][1]
inches_per_pixel = 1 / size_obj.pixelsPerMetric

img = cv2.imread(image,0)
fft = FFTObj(image, inches_per_pixel)
graph = fft.get_fft_graph()
plt.subplot(222),plt.plot(graph)
plt.title('FFT'), plt.xticks([]), plt.yticks([])
# plt.show()

peaks = fft.get_peaks(graph)
plt.subplot(224),plt.plot(peaks, len(peaks) * [1], "x")
plt.title('Peaks'), plt.xticks([]), plt.yticks([])
plt.show()

pitch = fft.get_pitch()

print('Inches per pixel: ' + str(inches_per_pixel))
print('Screw length: ' + str(length))
print('Thread pitch: ' + str(pitch))

plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Image'), plt.xticks([]), plt.yticks([])

plt.subplot(222),plt.plot(graph)
plt.title('FFT'), plt.xticks([]), plt.yticks([])

plt.subplot(224),plt.plot(peaks, len(peaks) * [1], "x")
plt.title('Peaks'), plt.xticks([]), plt.yticks([]), plt.xlabel("Calculated Pitch: " + str(fft.get_pitch()) + " in")

plt.show()