<H1> ADAS ML Inference System  </H1>

This project is a FastAPI-based backend that processes vehicle telemetry data and ppredicts accident risk using a trained ML classification model.

It exposes REST APIs for real-time inference and is tested using Swagger UI.

Example:
Input: telemetry data (speed, braking, steering)
Output: risk score (low / medium / high)
<br>

<H2>Architecture Diagram</H2>

Vehicle Telemetry Input

   ↓

FastAPI Backend (Inference Service)

   ↓

Feature Processing

   ↓

ML Model (Risk Prediction)

   ↓

API Response (Risk Score)

   ↓

Swagger UI (Testing)

<H2> How to Run the Project </H2>

Follow the steps below in order to run the ADAS ML Platform Mini end-to-end.

Prerequisites

Make sure you have:

Python 3.9+
pip installed

Install dependencies:

       pip install -r requirements.txt

## Dataset Setup

Place your dataset in the following structure:

                         data/raw/
                         ├── safe/
                         ├── caution/
                         └── dangerous/
#### Each folder should contain corresponding images.

## Execution Steps
### Step 1 — Validate Dataset

    python run_validate.py

Purpose: Verifies dataset integrity before training
Detects: Missing class folders
Corrupted images
Class distribution

#### Note: Run this once (or whenever dataset changes)
<br>

### Step 2 — Train Model

    python -m app.model.train

Purpose:

Trains the model using the prepared dataset
Uses transfer learning (ResNet18)

Output:

    artifacts/model/model.pth
#### Important: This step is mandatory before evaluation or inference
<br>

### Step 3 — Evaluate Model
    python -m app.model.evaluate

Purpose: Evaluates model performance on test data
Computes:
-> Accuracy
-> Precision
-> Recall
-> F1 Score
-> Confusion Matrix

Output:

    artifacts/metrics/metrics.json
<br>

### Step 4 — Run API (FastAPI)
    uvicorn app.api.main:app --reload

Purpose: Starts inference API server

Access API Docs (Swagger UI):
         
    http://127.0.0.1:8000/docs
#### Note: Keep this terminal running   
<br>

# It will Show the Result like this
### > Model predicted Danger
<img width="1511" height="948" alt="image" src="https://github.com/user-attachments/assets/e28414bc-266e-4c3f-b875-e0e85f0ba83d" />
<br> <br>

### > Model predicted to be Cautious
<img width="1511" height="948" alt="image" src="https://github.com/user-attachments/assets/b3cdb624-32f6-465c-a74c-232d6d338a9f" />
<br> <br>


### > Model predicted Safety
<img width="1511" height="948" alt="image" src="https://github.com/user-attachments/assets/3990db76-2f25-4a0e-accd-486d43f91eef" />
<br> <br> <br>


### Step 5 — Run Dashboard (Streamlit)

Open a new terminal window, then run:

    streamlit run dashboard/streamlit_app.py

Purpose: Visualizes model performance and metrics

Access Dashboard:

    http://localhost:8501

## Execution Flow Summary
    Validate → Train → Evaluate → Serve API → View Dashboard    

<img width="1463" height="856" alt="image" src="https://github.com/user-attachments/assets/805a5b67-a9cb-40f7-b2b2-34cd3cc13f4a" />



