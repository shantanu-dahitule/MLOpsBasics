# MLOps Project with ZENML

## Overview
This project demonstrates the implementation of an MLOps pipeline for predicting reviews using ZENML. ZENML is used to create a machine learning pipeline, making it easier to manage and reproduce the entire machine learning workflow.

## Prerequisites
Ensure you have the following installed before running the project:
- Python (>=3.8)
- ZENML (`pip install zenml`)
- Other dependencies specified in `requirements.txt`

## Project Structure
- `data/`: Contains the dataset for training and testing.
- `src/`: Holds the source code for the machine learning model.
- `steps/`: Stores the ZENML pipeline configurations.

#### Note: Use bash terminal or Unix terminal for successful execution.
## Getting Started
Clone the repository:
   ```bash
   git clone https://github.com/shantanu-dahitule/MLOpsBasics.git
   cd MLOpsBasics
   pip install -r requirements.txt
   zenml init
   zenml run
To see the dashboard use
   zenml up --blocking

