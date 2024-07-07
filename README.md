# Signature_Attestation

This project is a Flask-based web application for signature verification and analysis. It uses OpenCV for image processing and feature extraction, and allows users to upload two images of signatures for comparison. The system analyzes the signatures for similarity, stroke velocity, and pressure.

## Features

-> Upload Signatures: Users can upload two signature images for comparison.
-> Preprocessing: Converts images to grayscale, resizes them, and applies binary thresholding.
-> Feature Extraction: Uses ORB (Oriented FAST and Rotated BRIEF) to extract and match features between the two images.
-> Match Visualization: Displays the matched features between the two signatures.
-> Stroke Analysis: Extracts strokes and estimates velocity and pressure based on the contours of the signature

## Installation

### Clone the repository:
     git clone https://github.com/GaddamVarshith/Signature_Attestation.git
     cd signature-Attestation

### Create a virtual environment and activate it:
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`

### Install the dependencies:
  pip install -r requirements.txt
  
### Ensure the uploads and static directories exist:
mkdir -p uploads static

## Usage
Run the Flask application:
flask run
Open your web browser and navigate to http://127.0.0.1:5000.

Upload two signature images for comparison.

## File Structure
app.py: The main Flask application.
templates/: HTML templates for the web pages.
upload.html: Form for uploading signature images.
result.html: Displays the results of the signature comparison and analysis.
uploads/: Directory for storing uploaded images.
static/: Directory for storing result images.
## Functions
`preprocess_image
give this all information github readme text format like #
ChatGPT
Signature Verification and Analysis System
This project is a Flask-based web application for signature verification and analysis. It uses OpenCV for image processing and feature extraction, and allows users to upload two images of signatures for comparison. The system analyzes the signatures for similarity, stroke velocity, and pressure.

## Features
Upload Signatures: Users can upload two signature images for comparison.
Preprocessing: Converts images to grayscale, resizes them, and applies binary thresholding.
Feature Extraction: Uses
  

