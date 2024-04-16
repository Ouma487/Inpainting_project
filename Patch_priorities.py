import numpy as np
import skimage.morphology as morpho  
from skimage import img_as_float
import matplotlib.pyplot as plt
import cv2
import torch



def get_patch(image,p,patch_size):
    """
    Returns a patch centered on p
    """
    r = patch_size//2
    clip = np.array(image[p[0]-r:p[0]+r+1,p[1]-r:p[1]+r+1])
    return clip 


def similarity(patch1 , patch2 , maskpatch):
    d=0
    for i in range(3): 
        d+= np.sum(maskpatch*(patch1[:,:,i]-patch2[:,:,i])**2)
    return d




def confidence(mask,patch_size):
    """
    Compute the confidence map based on the provided mask.

    Parameters:
        mask (numpy.ndarray): The mask to be used for the computation.

    Returns:
        numpy.ndarray: The confidence map.
    """
    n, m = mask.shape
    c=np.zeros(mask.shape)
    for k in range(n):
        for l in range(m):
            patch = get_patch(mask,(k,l),patch_size)
            c[k,l]=np.sum(patch)/(patch.shape[0]*patch.shape[1])
    return c


def gradient_I(image, mask, bordure,patch_size):
    """
    Compute the gradient I(p) for all p in delta(\Omega).

    Parameters:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The binary mask indicating the filled and unfilled regions.
        bordure (list): List of border pixel coordinates.

    Returns:
        list: List containing the x and y components of the gradient I(p) for each point in the border.
    """
    h, w = image.shape[:2]
    c = image.copy()
    c = c.astype(np.float64)
    c[mask == 0] = np.NaN  # Mask the outside of the mask
    
    # Separate color channels
    c_r, c_g, c_b = img_as_float(c[:, :, 0]), img_as_float(c[:, :, 1]), img_as_float(c[:, :, 2])
   
    fgradx, fgrady = np.zeros((h, w)), np.zeros((h, w))
    
    for point in bordure:
        # Extract patches for each channel outside the loop
        patches_r = get_patch(c_r, point, patch_size)
        patches_g = get_patch(c_g, point,patch_size)
        patches_b = get_patch(c_b, point,patch_size)
        
        # Compute gradients for each channel
        gradients_r = np.nan_to_num(np.gradient(patches_r))
        gradients_g = np.nan_to_num(np.gradient(patches_g))
        gradients_b = np.nan_to_num(np.gradient(patches_b))
        
        # Compute norme for each channel
        norme_r = np.sqrt(gradients_r[0]**2 + gradients_r[1]**2)
        norme_g = np.sqrt(gradients_g[0]**2 + gradients_g[1]**2)
        norme_b = np.sqrt(gradients_b[0]**2 + gradients_b[1]**2)
        
        # Compute maximum norme across channels
        norme = np.maximum(np.maximum(norme_r, norme_g), norme_b)
        
        # Find the maximum norme index for each patch
        max_patch = np.unravel_index(norme.argmax(), norme.shape)
        
        # Assign gradients based on the maximum norme index
        fgradx[point[0], point[1]] = gradients_r[0][max_patch]
        fgrady[point[0], point[1]] = gradients_r[1][max_patch]
    
    return [fgradx, fgrady]


def normal_vect(image, mask, bord,patch_size):
    """
    Compute the normal vectors at each point on the boundary of the mask.

    Parameters:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The binary mask indicating the filled and unfilled regions.
        bord (list): List of border pixel coordinates.

    Returns:
        tuple: Tuple containing the x and y components of the normal vectors for each point in the border.
    """
    h, w = mask.shape[:2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coordx, coordy = torch.zeros((h, w), dtype=torch.float, device=device), torch.zeros((h, w), dtype=torch.float, device=device)
    
    for p in bord:
        i, j = p
        patch = get_patch(mask,(i,j),patch_size)
        
        grad = len(patch)*len(patch[0])*np.nan_to_num(np.array(np.gradient(patch)))
        grad = torch.tensor(grad, dtype=torch.float, device=device)
        gradX, gradY = grad[0], grad[1]
        
        # Compute the central index of the patch
        centerX, centerY = patch.shape[0] // 2, patch.shape[1] // 2
        
        # Compute the x and y components of the normal vector
        coordx[i, j] = gradX[centerX, centerY]
        coordy[i, j] = gradY[centerX, centerY]
    
    return coordy.cpu().numpy(), coordx.cpu().numpy()



def P(image, mask, bordure,patch_size):
    """
    Computes P for points on the border.

    Parameters:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The binary mask indicating the filled and unfilled regions.
        bordure (list): List of border pixel coordinates.

    Returns:
        numpy.ndarray: The computed P values for each point on the border.
    """
    h, w = mask.shape[:2]
    P = np.zeros((h, w))
    C = confidence(mask,patch_size)
    I = gradient_I(image, mask, bordure,patch_size)
    N = normal_vect(image, mask, bordure,patch_size)

    # Extracting i and j coordinates from bordure
    i_coords, j_coords = zip(*bordure)

    # Compute P_values for all coordinates in bordure
    P_values = np.abs(I[0][i_coords, j_coords] * N[0][i_coords, j_coords] +
                      I[1][i_coords, j_coords] * N[1][i_coords, j_coords]) / 255 * C[i_coords, j_coords]

    # Assign computed values to P array at corresponding coordinates
    P[i_coords, j_coords] = P_values

    return P

def maxP(image,mask,bordure,patch_size):
    
    "Finds point with max value of P"
    
    p=P(image,mask,bordure,patch_size)
    maximum=p[bordure[0]]
    argmax=bordure[0]
    for point in bordure:
        i,j=point
        if(p[i][j]>=maximum):
            maximum = p[i][j]
            argmax=point
    return argmax  