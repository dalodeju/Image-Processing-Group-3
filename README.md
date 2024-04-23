# Flower Segmentation

## Student Details
### Names & IDs:
```
Darrel Loh De Jun           20414780
Samuel Joshua Anand         20497938
Keosha Vadhnii              20415775
Valencia Ann Raj Davidraj   20410418
```
### Group: 3


## Overview
The aim of this project is to develop a Python program capable of segmenting various flower regions in images through an image processing pipeline. The requirements state that the program is to process one image at a time with minimal user input, and produce a binary image that marks the regions corresponding to the flower material. The solution should be as automated as posible and must be able to operate on all images without user intervention.


## Compilation Instructions
### How to run
1. **Install Python**: Download and install Python from [python.org](https://www.python.org/).
2. **Install Required Libraries**: Use pip to install the necessary Python libraries:
   ```
   pip install opencv-contrib-python matplotlib numpy scikit-image
   ```


## Program Execution
1. Clone or download the project repository from this [link](https://github.com/dalodeju/Image-Processing-Group-3.git).
2. Navigate to the project directory.
3. Unzip flowerSegmentation.zip
4. Execute the Python program using the following command:
   ```
   python flowerSegmentation.py
   ```
5. The program will process each image in the 'input-images' folder, segment the flower regions, and save the binary segmentation results in the 'output' folder.
6. Intermediate images generated during the processing pipeline will also be saved in the 'image-processing-pipeline' folder.


## Evaluation
The proposed image processing pipeline adopts techniques including color space conversion, bias removal, thresholding and binary processing. Each layer of the process sharpens up the edges of flower and segments them from their background. This process to isolate flowers from their backgrounds makes use of LAB color space along with noise reduction and Otsu's thresholding method, and it results in binary representations for more operations to be done. Morphological processing operations refines the previous segmentation results and thus enrich of the flower outline. Finally, the evaluation conducted using structural similarity comparison between automated segmented images and ground truth annotations allows the best segmented images to be selected as the final result. 

## Conclusion
In conclusion, our developed image-processing pipeline offers an integrated approach to the flower segmentation problem that would be useful for botanical research , plant phenotyping, and crop farming. The automation of segmentation tasks provides the pipeline with the ability to quickly analyze large datasets. Additionally, it makes the output more consistent hence cutting back the error brought about by manual operations. The value of this work is its provision of an improved method to increase the effectiveness and precision of a floral analysis resulting in the enhancement of research and the applications that demand it.
