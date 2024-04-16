
import numpy as np
import skimage.morphology as morpho  
from skimage import img_as_float
import matplotlib.pyplot as plt
import cv2


# Loading & Displaying an image
def loadImagebis(src):
    img = cv2.imread(src, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(5,5))  # Create a new figure for each image
    plt.imshow(rgb, interpolation='nearest')
    plt.axis('off')
    plt.show()


def view_data(img, figsize=(5, 5)):
    """
    Display an image.

    Parameters:
        img (numpy.ndarray): The image data. It can be grayscale or RGB.
        figsize (tuple): The size of the figure (default is (5, 5)).

    Returns:
        None
    """
    plt.figure(figsize=figsize)
    
    # If it's a binary image (e.g., border matrix), convert it to RGB for visualization
    if img.ndim == 2:  # Check if the image is grayscale
        img = np.stack((img, img, img), axis=-1)  # Convert to RGB by duplicating the channel
    elif img.shape[2] == 1:  # Check if the image has only one channel
        img = np.repeat(img, 3, axis=2)  # Duplicate the single channel to create an RGB image
        
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# Mask manipulation
# Creating a rectangular mask
def mask(img, x1, x2, y1, y2):
    """
    Takes an image and four coordinates, produces a mask with the same size as the image.

    Parameters:
        img (numpy.ndarray): Input image.
        x1, x2, y1, y2 (int): Coordinates defining the rectangle to be masked.

    Returns:
        numpy.ndarray: Mask with the specified region set to 0.
    """
    height, width = img.shape[:2]
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    # Create a mask with ones
    mask = np.ones((height, width), dtype=int)

    # Set the specified rectangle region to 0
    mask[y1:y2, x1:x2] = 0

    return mask


# Reading a mask
def read_mask(mask):
    """
    Writes the provided mask to a PNG file named "mask.png".

    Parameters:
        mask (numpy.ndarray): The mask to be saved as an image.

    Returns:
        None
    """
    cv2.imwrite("mask.png", mask)

# Removing the masked area from the image
def delete_zone(img, mask):
    """
    Remove the masked area from the input image.

    Parameters:
        img (numpy.ndarray): The input image.
        mask (numpy.ndarray): The mask indicating the area to be removed.

    Returns:
        numpy.ndarray: The image with the masked area removed.
    """
    img = img_as_float(img)
    new_img = np.copy(img)
    new_img[mask == 0] = 0
    
    # Convert to 8-bit unsigned integer image
    new_img = (new_img * 255).astype(np.uint8)

    # Convert BGR to RGB
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    
    return new_img



def init_bord_m(mask):
    """
    Initialize the border of a mask.

    Parameters:
        mask (numpy.ndarray): Input binary mask.

    Returns:
        list: List of border pixel coordinates.
    """
    # Apply morphological operation to find the borders
    bords = mask - morpho.erosion(mask)
    
    # Find the coordinates of the border pixels
    border_coords = np.argwhere(bords == 1)
    
    # Convert coordinates to a list of tuples
    L = [(i, j) for i, j in border_coords]
    
    return L


def display_mask_with_border(masked_image, border_coordinates, figsize=(8, 6)):
    """
    Display the image with the mask overlaid and the border highlighted.

    Parameters:
        masked_image (numpy.ndarray): The image with the mask overlaid.
        border_coordinates (list): List of border pixel coordinates.
        figsize (tuple): Size of the figure. Default is (8, 6).

    Returns:
        None
    """
    # Create a new figure
    plt.figure(figsize=figsize)
    
    # Display the image with the mask overlaid
    plt.imshow(masked_image)
    
    # Highlight the border pixels
    for coord in border_coordinates:
        plt.plot(coord[1], coord[0], 'yo', markersize=2)  # Yellow border pixels
    
    # Turn off axis
    plt.axis('off')
    
    # Show the plot
    plt.show()
