import cv2
import os
from tkinter import filedialog
from morphoglia import set_path

# Initialize global variables
drawing = False
value = 1  # Default to brush drawing
brush_size = 3  # Default brush size
draw_color = (0, 0, 0)  # Default to black

def draw_circle(event, x, y, flags, param):
    global drawing, value, brush_size, rect_start, image, draw_color

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rect_start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if value == 0:  # Rectangle mode
            # Drawing logic for rectangles handled in LBUTTONUP
            pass
        elif value == 1:  # Brush drawing mode
            cv2.circle(image, (x, y), brush_size, draw_color, -1)
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        if value == 0:
            cv2.rectangle(image, rect_start, (x, y), draw_color, -1)  # Draw rectangle on mouse release

if __name__ == "__main__":
    input_path = filedialog.askdirectory()
    o_path = set_path(os.path.join(input_path, "Edited_images"))

    for filename in os.listdir(input_path):
        base, extension = os.path.splitext(filename)
        if base.endswith("_labeled") or extension.lower() not in {".jpg", ".png", ".tif"}:
            continue

        input_file = os.path.join(input_path, filename)
        image = cv2.imread(input_file, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: Unable to load the image {filename}.")
            continue

        cv2.namedWindow(f"Image: {filename}")
        cv2.setMouseCallback(f"Image: {filename}", draw_circle)

        while True:
            cv2.imshow(f"Image: {filename}", image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                value = 0  # Rectangle mode
            elif key == ord('c'):
                value = 1  # Brush drawing mode
            elif ord('1') <= key <= ord('9'):
                brush_size = key - ord('0')
            elif key == ord('b'):
                draw_color = (0, 0, 0)  # Set draw color to black
            elif key == ord('f'):
                draw_color = (255, 255, 255)  # Set draw color to white
            elif key == 27:
                break  # Exit on Esc key

        # Save the edited image
        output_path = os.path.join(o_path, f"{base}.tif")
        cv2.imwrite(output_path, image)
        cv2.destroyAllWindows()
