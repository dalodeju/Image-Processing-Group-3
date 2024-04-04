# Flower Segmentation
# Group 3

import cv2
import os
from skimage.metrics import structural_similarity as ssim


# Function for binary image processing (e.g., morphological operations)

def convert_to_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


# Function for OTSU thresholding
def otsu_thresholding(image, threshold=150):
    _, segmented_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return segmented_image


def binary_processing(image):
    # Apply morphological operations (e.g., dilation, erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    processed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return processed_image


# Function for noise reduction
def noise_reduction(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)


# Function for post-processing to select best segmented image based on ground truth
def post_processing(segmented_image, ground_truth_image):
    # Ensure both images are not None

    # Resize images to have the same dimensions
    ground_truth_grayscale = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)
    ground_truth_resized = cv2.resize(ground_truth_grayscale, (segmented_image.shape[1], segmented_image.shape[0]))

    # Calculate structural similarity
    similarity_score, _ = ssim(segmented_image, ground_truth_resized, full=True)
    return similarity_score


# Modify flower_segmentation function to incorporate binary processing and noise reduction
def flower_segmentation(image, pipeline_folder, index):
    lab_image = convert_to_lab(image)
    L, A, B = cv2.split(lab_image)

    # Define image processing pipeline folder locations for each input image
    pipeline_folder_output = f'{pipeline_folder}{index + 1}/'  # Adjusted indexing here
    L_out = pipeline_folder_output + 'L/'
    A_out = pipeline_folder_output + 'A/'
    B_out = pipeline_folder_output + 'B/'

    # writing the initial L.A.B images to the image processing pipeline
    cv2.imwrite(os.path.join(L_out, '1_L_out.jpg'), L)
    cv2.imwrite(os.path.join(A_out, '1_A_out.jpg'), A)
    cv2.imwrite(os.path.join(B_out, '1_B_out.jpg'), B)

    L_median = noise_reduction(L)
    A_median = noise_reduction(A)
    B_median = noise_reduction(B)

    # writing the median L.A.B images to the image processing pipeline
    cv2.imwrite(os.path.join(L_out, '2_L_median.jpg'), L_median)
    cv2.imwrite(os.path.join(A_out, '2_A_median.jpg'), A_median)
    cv2.imwrite(os.path.join(B_out, '2_B_median.jpg'), B_median)

    L_segmented = otsu_thresholding(L_median)
    A_segmented = otsu_thresholding(A_median)
    B_segmented = otsu_thresholding(B_median)

    # writing the equalized L.A.B images to the image processing pipeline
    cv2.imwrite(os.path.join(L_out, '3_L_segmented.jpg'), L_segmented)
    cv2.imwrite(os.path.join(A_out, '3_A_segmented.jpg'), A_segmented)
    cv2.imwrite(os.path.join(B_out, '3_B_segmented.jpg'), B_segmented)

    # Binary image processing
    L_processed = binary_processing(L_segmented)
    A_processed = binary_processing(A_segmented)
    B_processed = binary_processing(B_segmented)

    cv2.imwrite(os.path.join(L_out, '4_L_processed.jpg'), L_processed)
    cv2.imwrite(os.path.join(A_out, '4_A_processed.jpg'), A_processed)
    cv2.imwrite(os.path.join(B_out, '4_B_processed.jpg'), B_processed)

    return L_processed, A_processed, B_processed


## Define input image paths and output folder prefix
# input images path
easy_input_paths = ['dataset/input_images/easy/easy_1.jpg', 'dataset/input_images/easy/easy_2.jpg',
                    'dataset/input_images/easy/easy_3.jpg']
medium_input_paths = ['dataset/input_images/medium/medium_1.jpg', 'dataset/input_images/medium/medium_2.jpg',
                      'dataset/input_images/medium/medium_3.jpg']
hard_input_paths = ['dataset/input_images/hard/hard_1.jpg', 'dataset/input_images/hard/hard_2.jpg',
                    'dataset/input_images/hard/hard_3.jpg']
# output folder (not final output)
easy_pipeline_folder = 'imageprocessing-pipeline/easy/easy'
medium_pipeline_folder = 'imageprocessing-pipeline/medium/medium'
hard_pipeline_folder = 'imageprocessing-pipeline/hard/hard'


# Modify main function to include post-processing
def main():
    global pipeline_folder
    input_folder = "dataset/input_images/"
    # path for final outputs
    output_folder = "output/"
    # path for ground truth images
    ground_truth_folder = "dataset/ground_truths/"

    for difficulty in ["easy", "medium", "hard"]:
        input_subfolder = os.path.join(input_folder, difficulty)
        output_subfolder = os.path.join(output_folder, difficulty)
        ground_truth_subfolder = os.path.join(ground_truth_folder, difficulty)

        # Get input image paths
        input_paths = [os.path.join(input_subfolder, filename) for filename in os.listdir(input_subfolder) if filename.endswith(".jpg") or filename.endswith(".jpeg")]

        # Get ground truth image paths
        ground_truth_paths = [os.path.join(ground_truth_subfolder, filename.replace(".jpg", ".png")) for filename in os.listdir(input_subfolder) if filename.endswith(".jpg") or filename.endswith(".jpeg")]

        # Select the appropriate pipeline folder based on difficulty level
        if difficulty == "easy":
            pipeline_folder = easy_pipeline_folder
        elif difficulty == "medium":
            pipeline_folder = medium_pipeline_folder
        elif difficulty == "hard":
            pipeline_folder = hard_pipeline_folder

        # Initialize index
        index = 0

        for input_path, ground_truth_path in zip(input_paths, ground_truth_paths):
            # Create output folder for current image
            filename = os.path.splitext(os.path.basename(input_path))[0]
            output_image_folder = os.path.join(output_subfolder, filename)
            os.makedirs(output_image_folder, exist_ok=True)

            # Read input image
            img = cv2.imread(input_path)

            # Apply flower segmentation pipeline with correct index
            L_segmented, A_segmented, B_segmented = flower_segmentation(img, pipeline_folder, index)

            # Increment index for next iteration
            index += 1

            # Save segmented images
            cv2.imwrite(os.path.join(output_image_folder, "L_segmented.jpg"), L_segmented)
            cv2.imwrite(os.path.join(output_image_folder, "A_segmented.jpg"), A_segmented)
            cv2.imwrite(os.path.join(output_image_folder, "B_segmented.jpg"), B_segmented)

            # Load corresponding ground truth image
            ground_truth_img = cv2.imread(ground_truth_path)
            if ground_truth_img is None:
                print(f"Error: Unable to load ground truth image: {ground_truth_path}")
                continue  # Skip to the next image

            # Post-processing to select the best segmented image
            value1 = post_processing(L_segmented, ground_truth_img)
            value2 = post_processing(A_segmented, ground_truth_img)
            value3 = post_processing(B_segmented, ground_truth_img)

            # Choose the best segmented image based on similarity scores
            if value3 < value2:
                if value3 < value1:
                    best_segmented_image = B_segmented
                else:
                    best_segmented_image = L_segmented
            else:
                if value2 < value1:
                    best_segmented_image = A_segmented
                else:
                    best_segmented_image = L_segmented

            # Save the best segmented image
            # print the similarity scores for each of the segmented images
            print(value1, value2, value3)
            #save the best segmented image under the output folder for each image
            cv2.imwrite(os.path.join(output_image_folder, "best_segmented_image.jpg"), best_segmented_image)


if __name__ == "__main__":
    main()
