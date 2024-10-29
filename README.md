# Bubble Converage Tracker
## Quick start
> conda create env -n Unet
> conda activate Unet
> git clone 

## Background
As human civilization advances, global water pollution has caused significant economic losses and impacted daily life. Our aim is to develop on-site monitoring technology to detect pollutant types using:
- Optical and image recognition
- Non-contact measurement for high efficiency and accuracy

## Motivation
In sewage treatment plants, foam control typically relies on chemical agents, but determining the correct dosage is challenging:
- **Underdosing** fails to inhibit foam.
- **Overdosing** harms microorganisms and may still not control foam.
  
An accurate real-time monitoring system is essential to automate analysis and improve foam treatment efficiency.

## Solution Overview

### Optical Component
- **Time-of-Flight (TOF) Sensors**: Measure foam height and liquid levels.
- **UV Absorption Spectra**: Monitor pollutants (e.g., BSA, SDBS).
- **Regression Analysis**: Confirm linear relationships between concentration and absorbance.

### Image Component

#### Image Acquisition and Pre-processing
- Capture foam features using a camera.
- Apply U-Net segmentation for foam spread analysis.

#### U-Net Image Segmentation
- U-Net, a modified autoencoder with a U-shaped structure, excels at semantic segmentation:
  - More precise than traditional object detection.
  - Widely used in medical and satellite imaging.

![Unet](https://github.com/user-attachments/assets/2431ea1b-29de-44f1-a086-1169e360e186)

#### Implementation with Labelme
- Use Labelme for polygon-based foam annotations.
- Convert JSON labels into mask images (foam vs. background).

![bubble](https://github.com/user-attachments/assets/30b57e1a-1445-4c60-bfbc-27b4b50a2765)


#### Training and Prediction
- Train U-Net with labeled data to generate masks.
- Use predicted masks to identify foam regions and measure spread in real-time.

## Results Summary
- **Optical Component**: Successfully measured foam height and liquid levels.
- **Image Component**: Demonstrated U-Net segmentation's necessity for enhancing recognition accuracy in complex environments.

## Future Applications
- Integrate U-Net segmentation for mixed foam concentrations.
- Further develop the model to improve real-time accuracy in wastewater treatment systems.

