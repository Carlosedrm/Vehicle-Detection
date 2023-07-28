import cv2 as cv
import numpy as np

# The video
video = cv.VideoCapture('highway1.mp4')

# Check if there's a video
if (video.isOpened()== False):
    print("Não foi possivel abrir o video")

subtractor = cv.bgsegm.createBackgroundSubtractorMOG()
# The count of how many cars there are
count = 0

# For the size of the contours
min_contour_size = 60
max_contour_size = 460

# The line
line_y_1 = 460
line_y_2 = 400

# The buffer
crossed_buffer = 5

# Things needed to save the output video
output_file = "output_video2.mp4"
output_fps = 30  # You can adjust the FPS as needed
output_size = (800, 600)
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for saving as MP4 file
output_video = cv.VideoWriter(output_file, fourcc, output_fps, output_size)

while(video.isOpened()): 
    # Capture frame-by-frame
    ret, frame = video.read()
    if ret == True:       
        frame = cv.resize(frame, (800, 600))

        # Turn the frame into a gray image
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  

        # Blur the image a little bit
        blurred = cv.GaussianBlur(gray, (5, 5), 5)

        # Make the image a binary one with mog
        bin = subtractor.apply(blurred)

        # Closing the image
        dilate = cv.dilate(bin, np.ones((5,5)))
        erode = cv.erode(dilate, np.ones((5,5)))
        
        # Showing the binary version
        cv.imshow('Frame', erode)

        # Find the contours
        contours, _ = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            # Check the size of the contours so that the small and big ones are ignored
            cnt = (min_contour_size <= w <= max_contour_size) and (min_contour_size <= h <= max_contour_size)
            if not cnt:
                continue
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a rectangle around the detected car
            
            # Calculate the centroid of the contour
            centroid_x = x + w // 2
            centroid_y = y + h // 2
            cv.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Draw a red circle at the centroid

            # Check if the car is near where the line is
            # We need the +6 and -6 because the car's might not be exactly where the line is
            if (centroid_y + crossed_buffer) > line_y_1 > (centroid_y - crossed_buffer):
                count += 1

        # Display the count as text on the frame and draw the line
        cv.putText(frame, f"Count: {count}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #The line for the first video
        line1 = cv.line(frame, (20, 460), (700, 460), (0, 255, 0), 3)
        #line2 = cv.line(frame, (200, 400), (750, 400), (0, 255, 0), 3)

        # Write the frame to the output video
        output_video.write(frame)
        
        # The final video
        cv.imshow('Video', frame)
        # Press Q on keyboard to exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
output_video.release()

# Closes all the frames
cv.destroyAllWindows()
print("The number of vehicles is: ", count)