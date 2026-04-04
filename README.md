<H2> ADAS ML Inference System </H2>

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
