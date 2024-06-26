# Exemplar-Based Inpainting for Object Removal and Image Restoration

This repository contains the implementation and application of an exemplar-based inpainting algorithm for object removal and image restoration, as part of a computer vision project.

## Directory Structure

- `iter_algo.py`: This script contains the main iteration logic of the exemplar-based inpainting algorithm.

- `tools.py`: This utility script includes functions for displaying images, creating masks, and identifying the borders of masks.

- `patch_priorities.py`: A Python file dedicated to computing the priority of patches during the inpainting process. The patch priority is based on the confidence and data terms, as explained by the following equations:

    ### Patch Priority Function
    The priority of a patch $P(p)$ is calculated as:
    $$P(p) = C(p) \times D(p)$$

    #### Confidence Term \(C(p)\)
    The confidence term represents the amount of reliable information in a patch and is given by:
    $$C(p) = \frac{\sum_{i \in \psi_p} C(i)}{|\psi_p|}$$
    where $|\psi_p|$ is the area of the patch.

    #### Data Term $D(p)$
    The data term promotes patch filling in areas with strong edges, calculated using the image gradient at the patch border:
    $$D(p) = | \nabla I_p \cdot n_p |$$
    where $\nabla I_p$ is the image gradient and $n_p$ is the normal to the patch boundary.

-  `Synthetic_images.ipynb`: A Jupyter notebook demonstrating the application of the inpainting method to synthetic images.
  
![Synthetic Images](Images/BW_RGB.jpg)

- `Bouregreg.ipynb`: Application of the inpainting algorithm to the Bouregreg image.
- `Hassan_&_Zellij.ipynb`: Application of the inpainting algorithm to the Hassan & Zellij image.
- `Kasbah_of_Udaya.ipynb`: Application of the inpainting algorithm to the Kasbah of Udaya image.
  
![Images of landmarks and patterns](Images/images.jpg)


- `Exemplar_Based_Inpainting_for_Object_Removal_and_Image_Restoration.pdf`: A detailed report summarizing the methods used and the results achieved.

- `LICENSE`: The GNU General Public License v3.0 under which this project is licensed.

## How to Use

To use the inpainting scripts, you will need to have Python installed along with the necessary dependencies such as OpenCV and Matplotlib. Run the scripts or notebooks to apply the inpainting algorithm to the provided images or your own.

## Contributing

Contributions to improve the code or the documentation are welcome. Please feel free to fork the repository and submit a pull request.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The project is based on the method pioneered by Criminisi et al., Object Removal by Exemplar Based Inpaiting paper.
- Thanks to the computer vision community for providing insights and valuable discussions that helped shape this project.
