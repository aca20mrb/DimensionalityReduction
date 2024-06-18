from PIL import Image

# Specify the path to your image
image_path = 'images/preprocessed_images/11_left.jpg'

# Load the image
image = Image.open(image_path)

# Display the image
image.show()