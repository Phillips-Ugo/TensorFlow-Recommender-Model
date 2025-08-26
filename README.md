# TensorFlow Student Recommender Model

This repository contains a TensorFlow-based recommender system for suggesting the best-fit classes to students based on their academic background, interests, and career goals.

## Features
- Uses TensorFlow and TensorFlow Recommenders for deep learning-based retrieval.
- Handles categorical, numerical, and multi-hot encoded features.
- Trains a model to recommend classes tailored to individual students.
- Example code for making predictions and saving models.

## Project Structure
```
recommender_model.py         # Main model and training script
saved_recommender_model/    # Directory for saved TensorFlow model
    fingerprint.pb
    saved_model.pb
    assets/
    variables/
        variables.data-00000-of-00001
        variables.index
student_data.csv            # (You must provide this file with your student data)
```

## Getting Started
1. **Install dependencies:**
   ```powershell
   python -m pip install tensorflow tensorflow-recommenders pandas numpy
   ```
2. **Prepare your data:**
   - Place a `student_data.csv` file in the project directory. Ensure columns match those referenced in `recommender_model.py`.
3. **Train the model:**
   - Run the script:
     ```powershell
     python recommender_model.py
     ```
4. **View recommendations:**
   - The script prints top recommended classes for a sample student.

## Customization
- Modify `student_data.csv` to fit your institution's data.
- Adjust model architecture in `recommender_model.py` for your needs.

## Requirements
- Python 3.8+
- TensorFlow 2.x
- tensorflow-recommenders
- pandas
- numpy

## License
MIT

## Author
Your Name
