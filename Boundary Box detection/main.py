import math

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def extract_lip_region(image_path, nose_point, cheek_point, yaw_angle, pitch_angle, roll_angle):
    # Load the image
    img = Image.open(image_path)

    # Calculate the midpoint between the two points
    midpoint = ((nose_point[0] + cheek_point[0]) / 2, (nose_point[1] + cheek_point[1]) / 2)

    # Calculate the distance between the two points
    distance = math.sqrt((cheek_point[0] - nose_point[0])**2 + (cheek_point[1] - nose_point[1])**2)

    # Calculate the radius of the circle based on the distance between the nose and cheek points
    radius = distance / 2

    # Calculate the center of the circle based on the yaw, pitch, and roll angles
    radians_yaw = math.radians(yaw_angle)
    radians_pitch = math.radians(pitch_angle)
    radians_roll = math.radians(roll_angle)

    # Calculate the direction vector of the camera based on the yaw, pitch, and roll angles
    direction_vector = (
        math.sin(radians_yaw) * math.cos(radians_pitch),
        math.sin(radians_pitch),
        math.cos(radians_yaw) * math.cos(radians_pitch)
    )

    # Calculate the axis of rotation for the roll angle
    roll_axis = (
        math.sin(radians_yaw),
        math.cos(radians_yaw),
        0
    )

    # Calculate the axis of rotation for the pitch angle
    pitch_axis = (
        -math.cos(radians_yaw) * math.sin(radians_pitch),
        math.sin(radians_yaw) * math.sin(radians_pitch),
        math.cos(radians_pitch)
    )

    # Calculate the center of the circle based on the direction vector, roll axis, and pitch axis
    center = (
        midpoint[0] - direction_vector[0] * radius,
        midpoint[1] - direction_vector[1] * radius,
        radius * math.sin(radians_roll)
    )
    margin = 20
    # Define the polygon boundary that covers the mouth and lips
    polygon_points = [
        (center[0] - radius, center[1] + radius / 2 - margin),
        (center[0] + radius, center[1] + radius / 2- margin),
        (center[0] + radius * 0.75, center[1] + radius * 1.5),
        (center[0] + radius * 0.25, center[1] + radius * 1.5),
        (center[0] - radius * 0.25, center[1] + radius * 0.75),
        (center[0] - radius * 0.75, center[1] + radius * 0.75)
    ]

    # Create a new image with the same size as the original image
    new_img = Image.new("RGBA", img.size, (0, 0, 0, 0))

    # Create a drawing context for the new image
    draw = ImageDraw.Draw(new_img)

    # Draw the polygon on the new image
    draw.polygon(polygon_points, fill=(255, 0, 0))

    # Create a mask from the new image
    mask = new_img.convert("L")

    # Apply the mask to the original image
    masked_img = Image.composite(img, new_img, mask)

    # Extract the lip region from the masked image
    lip_region = masked_img.crop((min(polygon_points, key=lambda x: x[0])[0],
                                  min(polygon_points, key=lambda x: x[1])[1],
                                  max(polygon_points, key=lambda x: x[0])[0],
                                  max(polygon_points, key=lambda x: x[1])[1]))
    plt.imshow(lip_region)
    plt.show()
    # Save the lip region image
    lip_region.save("lip_region.png")


extract_lip_region(
    r"C:\Users\ali\PycharmProjects\Lip_Movement_Detection\imgs_masked\2_4_Dancing_Dancing_4_355_0_surgical.png"
    , (112 * 0.3746092869685246, 112 * 0.8562057201678936), (112 * 0.9225053053635818, 112 * 0.7173771491417518),
    0.69823775, 0.3321294777777778, 0.3593641166666667)
