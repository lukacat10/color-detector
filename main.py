import cv2
import numpy as np
import datetime

input_video_file = "input.mp4"
output_video_file = "output.mp4"
output_timeline_file = "output_timeline.txt"

# Define the color ranges for red, yellow, and green
LOWER_RED = np.array([0, 100, 100])
UPPER_RED = np.array([10, 255, 255])

LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([30, 255, 255])

LOWER_GREEN = np.array([40, 100, 100])
UPPER_GREEN = np.array([90, 255, 255])

# Define the ROI coordinates (x, y, width, height)
roi_x, roi_y, roi_width, roi_height = 700, 500, 200, 200


def detect_traffic_light_color(roi_frame):
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV frame to get only the specific color range
    mask_red = cv2.inRange(hsv_frame, LOWER_RED, UPPER_RED)
    mask_yellow = cv2.inRange(hsv_frame, LOWER_YELLOW, UPPER_YELLOW)
    mask_green = cv2.inRange(hsv_frame, LOWER_GREEN, UPPER_GREEN)

    # Find contours in the masks
    contours_red, _ = cv2.findContours(
        mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_yellow, _ = cv2.findContours(
        mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_green, _ = cv2.findContours(
        mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Determine the color based on the largest contour area
    color = None
    if len(contours_red) > 0:
        color = "Red"
    elif len(contours_yellow) > 0:
        color = "Yellow"
    elif len(contours_green) > 0:
        color = "Green"
    else:
        color = "No traffic light detected or unknown color"

    return color


# Open video file
cap = cv2.VideoCapture(input_video_file)

# Get the video frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"H264")
out = cv2.VideoWriter(
    output_video_file, fourcc, frame_rate, (frame_width, frame_height)
)


# Define the previous color for comparison
prev_color = None

# Define start time
start_time = datetime.datetime.now()

# Open a file to save timeline events
timeline_file = open(output_timeline_file, "w")

while cap.isOpened():
    ret, frame = cap.read()
    if ret is True:
        # Crop the frame to focus on the ROI
        roi_frame = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]

        # Get the current color of the traffic light in the ROI
        color = detect_traffic_light_color(roi_frame)

        # Save the timeline events
        current_time = datetime.datetime.now()
        elapsed_time = current_time - start_time
        timeline_file.write(f"{elapsed_time}: {color}\n")

        # Display the color on the frame
        cv2.putText(
            frame,
            color,
            (roi_x, roi_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Draw a rectangle around the ROI
        cv2.rectangle(
            frame,
            (roi_x, roi_y),
            (roi_x + roi_width, roi_y + roi_height),
            (255, 0, 0),
            2,
        )

        # Write the frame
        out.write(frame)

        # Display the resulting frame
        cv2.imshow("frame", frame)

        # Store the current color for comparison in the next iteration
        prev_color = color

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

# Close the timeline file
timeline_file.close()
