
# Traffic-Sign-Detection-and-Recognition

An experimental model for the detection and recognition of road traffic signs in static images from roads of member states of the European Union. This model is based on **image processing** techniques and implemented using 'Matlab 2018b'.

## Project Overview

This project provides a solution to detect and recognize traffic signs from static images using color segmentation, shape detection, and template matching techniques. The primary goal of the project is to automate the process of identifying traffic signs in images, which can be useful for autonomous vehicles, traffic monitoring systems, and other similar applications.

## Key Technologies:
- **Color Segmentation:** To isolate the regions of interest (ROI) in the image based on traffic sign colors (e.g., red, blue, yellow).
- **Edge Detection & Shape Detection:** To identify common traffic sign shapes like circles, triangles, and rectangles.
- **Template Matching (Cross-Correlation):** To recognize specific traffic signs by comparing the detected ROI with a set of predefined templates.

## Features

- Detect traffic signs in static images based on their color and shape.
- Recognize the type of traffic sign by comparing detected shapes with templates.
- Uses basic image processing techniques (color segmentation and edge detection).
- MATLAB-based cross-correlation algorithm for template matching.

### Detection Pipeline

1. **Preprocessing:**
- Input images are converted to a suitable format for detection (e.g., color space conversion, image size).

2. **Color Segmentation:**
- The program segments the image based on the specific colors associated with traffic signs (e.g., red, blue, yellow). This helps narrow down regions of interest where traffic signs may be present.

3. **Edge Detection & Shape Detection:**
- The program applies edge detection (using methods like Canny or Sobel) to find the boundaries of objects in the image.
- Shape detection techniques are used to identify circular, triangular, or rectangular regions that correspond to common traffic sign shapes.

### Recognition Pipeline

1. **Template Matching (Cross-Correlation):**
- Once a region of interest is identified, the program uses a set of predefined templates for common traffic signs (e.g., stop sign, yield sign).
- The cross-correlation algorithm compares the detected shape with the templates and calculates a similarity score.
- The traffic sign is recognized based on the highest similarity match.

## Requirements

- MATLAB (R2018b or later)
- Image Processing Toolbox


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Vasilis-Thomas/Traffic-Sign-Detection-and-Recognition
    ```

2. Open MATLAB and navigate to the project directory:
    ```matlab
    cd('path_to_project_directory')
    ```

## Usage

1. **Prepare your images**: Place your static images for traffic sign detection in the `Images/` folder and specify the image you want to import.
    ```matlab
    img = imread('./Images/select_image');
    ```
   
2. **Run the detection script**:
    ```matlab
    run('Traffic_sign_detection_and_recognition.m');
    ```

3. Results will be displayed in MATLAB figure windows, showing the detection and recognition process.

4. You can modify the parameters (e.g., detection thresholds, classifier settings) in the main script to suit your dataset.

## Screenshots

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/2f795976-89aa-4788-a4b1-7357f15df207" alt="Original Image" width="200"/><br/>
        <b>Original Image</b>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/b1444c64-cba2-4c23-88c9-958c1411b06c" alt="Edge Detection (cleared image)" width="200"/><br/>
        <b>Edge Detection (cleared image)</b>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/c39211e0-bb8e-467d-8ce5-0a4c10cbe49e" alt="Edge Detection (with imfill)" width="200"/><br/>
        <b>Edge Detection (with imfill)</b>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/e158499d-4db9-4312-9f3d-0f86c1407d8a" alt="Boundaries" width="200"/><br/>
        <b>Boundaries</b>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/cbd8b8f8-dd13-458c-b699-455ba6a97ed0" alt="Color Mask (cleared image)" width="200"/><br/>
        <b>Color Mask (cleared image)</b>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/f8445c87-fd3e-4ace-8eec-2597f720e18c" alt="Color Mask (with imfill)" width="200"/><br/>
        <b>Color Mask (with imfill)</b>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/8a6da64c-ddd8-4860-b924-fca0381b7886" alt="Smoothed" width="200"/><br/>
        <b>Smoothed</b>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/f97f259d-8185-4da0-b50b-430f6901ba55" alt="Union" width="200"/><br/>
        <b>Union</b>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/281cd5e5-1fb5-4b7c-b186-b25ed808cc3e" alt="Intersection" width="200"/><br/>
        <b>Intersection</b>
      </td>
    </tr>
  </table>
</div>

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/7b7e4e67-8463-4390-ad27-02bbb618b70b" alt="Circularities" width="700"/><br/>
      </td>
    </tr>
    <tr>
      <td align="center">
        <br>
        <img src="https://github.com/user-attachments/assets/fb497b74-b9a1-43d1-b5ff-95251a29411b" alt="Possible Traffic Signs" width="500"/><br/>
        <b>Possible Traffic Signs</b>
      </td>
    </tr>
    <tr>
      <td align="center">
        <br>
        <img src="https://github.com/user-attachments/assets/33f1e237-f27e-470b-955a-9ea655d7c44e" alt="Recognized Traffic Signs" width="500"/><br/>
        <b>Recognized Traffic Signs</b>
      </td>
    </tr>
  </table>
</div>


## Authors

👤 **Θωμάς Βασίλειος**
* GitHub: [@Vasilis-Thomas](https://github.com/Vasilis-Thomas)

👤 **Σαρακενίδης Νικόλαος**
* GitHub: [@Nikoreve](https://github.com/Nikoreve)
