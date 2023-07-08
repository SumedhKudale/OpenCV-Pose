import cv2

protoFile = "/home/vboxuser/Downloads/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "/home/vboxuser/Downloads/pose/mpi/pose_iter_160000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

im = cv2.imread("/home/vboxuser/Desktop/File3.jpeg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
inWidth = im.shape[1]
inHeight = im.shape[0]

netInputSize = (368, 368)  # 368
inpBlob = cv2.dnn.blobFromImage(im, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
net.setInput(inpBlob)

# Forward Pass
output = net.forward()

# Number of keypoints
nPoints = 15

# Define POSE_PAIRS based on nPoints
POSE_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)]

# X and Y Scale
scaleX = inWidth / output.shape[3]
scaleY = inHeight / output.shape[2]

# Empty list to store the detected keypoints
points = []

# Threshold
threshold = 0.1

for i in range(nPoints):
    # Obtain probability map
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # Scale the point to fit on the original image
    x = scaleX * point[0]
    y = scaleY * point[1]

    if prob > threshold:
        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else:
        points.append(None)

imPoints = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)  # Convert back to BGR for drawing
imSkeleton = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)  # Convert back to BGR for drawing

# Draw points
for i, p in enumerate(points):
    cv2.circle(imPoints, p, 8, (255, 255, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.putText(imPoints, "{}".format(i), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, lineType=cv2.LINE_AA)

# Draw skeleton
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(imSkeleton, points[partA], points[partB], (255, 255, 0), 2)
        cv2.circle(imSkeleton, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

# Display images
cv2.imshow("Detected Keypoints", imPoints)
cv2.imshow("Skeleton", imSkeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()
