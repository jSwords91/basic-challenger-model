# Iris Classifier MLOps Basic Example

This project demonstrates some ML engineering techniques using scikit-learn, FastAPI, and Docker. It includes an Iris classifier model with API endpoints for prediction and training, as well as a challenger process for model improvement.

The model itself is almost an afterthought -- here we're thinking about the system. Insert whichever model you like.

I also set up Github Actions for CI/CD, where I run some basic tests upon push or pull request. This helps ensure that the code continues to work as expected.

This is totally artificial, but conveys some key ideas.

## Features

- FastAPI-based REST API
- Scikit-learn Iris classifier model
- Class-based model management
- Automatic model loading or training on startup
- Challenger process for model improvement
- Model versioning and persistence
- Logging for both application and model interactions
- Docker support for easy deployment

## Project Structure

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/iris-classifier.git
   cd iris-classifier
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

## Running the Application

### Using Python

1. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

2. Open your browser and navigate to `http://localhost:8000`

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t iris-classifier .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 iris-classifier
   ```

3. Open your browser and navigate to `http://localhost:8000`

## API Endpoints

- `POST /predict`: Make a prediction based on input features
- `GET /train`: Train the model and return the accuracy
