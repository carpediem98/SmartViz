from google.cloud import storage
import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils
from firebase_admin import messaging
from pyfcm import FCMNotification

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

    # Used as a size ref in the template
    quarter_diameter_inches = 0.955

    size_obj = SizeObj(img)
    size_obj.process(quarter_diameter_inches)

    length = size_obj.sizes[1][1]
    in_per_pixel = 1 / size_obj.pixelsPerMetric

    fft = FFTObj(img, in_per_pixel)
    pitch = fft.get_pitch()

    print("Processing file")
    print(f"Screw Length: {length}")
    print(f"Thread Pitch: {pitch}")

    # This registration token comes from the client FCM SDKs.
    registration_token = event['name'].split('/')[0]

    print(f"Token: {registration_token}")

    push_service = FCMNotification(api_key="AAAA6KZ1qeg:APA91bHbXB-yDl2ooRazu1rSDtng5k-6hmCrlt6TPAvdda6-BUCQ2VAbxJyR4OaaZ5SuWV2JyR3cbhymPap3nEXYTV3d9ZVuzxeemRzmtMCmrXLbC7_uT3KlPU_zR6_MdWisw9X68c_o")

    body = str(length) + ',' + str(pitch)
    result = push_service.notify_single_device(registration_id=registration_token, message_title='Results', message_body=body)

    # See documentation on defining a message payload.
    #message = messaging.Message(
    #    data={
    #        'screw_length': length,
    #        'thread_pitch': pitch,
    #    },
    #    token=registration_token,
    #)

    # Send a message to the device corresponding to the provided
    # registration token.
    #response = messaging.send(message)
    # Response is a message ID string.
    print('Successfully sent message:', result)


class FFTObj:

    def __init__(self, image, pixel_ratio):
        self.img = image
        self.pixel_ratio = pixel_ratio

    def get_fft_mag_spectrum(self):
        (thresh, img2) = cv2.threshold(self.img, 225, 255, cv2.THRESH_BINARY)
        #img2 = cv2.blur(img2, (1, 1280))
        f = np.fft.ifft2(img2)
        fshift = np.fft.fftshift(f)
        print(f)
        print(fshift)
        print(np.log(np.abs(fshift)))
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
        thresh = (max(graph) - min(graph)) * 0.009
        print(thresh)
        return find_peaks(graph, threshold=thresh)[0]

    def get_pitch(self):
        graph = self.get_fft_graph()
        peaks = self.get_peaks(graph)
        maxindex = graph.index(max(graph))
        peakindex = np.where(peaks == maxindex)[0][0]
        pixels = maxindex - peaks[peakindex - 1]
        return pixels * self.pixel_ratio


class SizeObj:

    def __init__(self, image_obj):
        # load the image, convert it to grayscale, and blur it slightly
        image = image_obj
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(image, (7, 7), 0)

        # perform edge detection, then perform a dilation + erosion to
        # close gaps in between object edges
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        (cnts, _) = contours.sort_contours(cnts)

        self.image = image
        self.contours = cnts
        self.pixelsPerMetric = None

    def process(self, width):
        sizes = []
        for c in self.contours:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 100:
                continue

            # compute the rotated bounding box of the contour
            orig = self.image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # if the pixels per metric has not been initialized, then
            # compute it as the ratio of pixels to supplied metric
            # (in this case, inches)
            if self.pixelsPerMetric is None:
                self.pixelsPerMetric = dB / width

            # compute the size of the object
            dimA = dA / self.pixelsPerMetric
            dimB = dB / self.pixelsPerMetric

            sizes.append((dimA, dimB))

        self.sizes = sizes


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)