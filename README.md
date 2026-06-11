# Enterprise Customer Churn Prediction Engine: End-to-End Deep Learning Architecture

An enterprise-grade, end-to-end deep learning framework designed to predict banking customer attrition (churn) with high calibration. This repository translates an abstract business challenge—customer retention—into a deterministic engineering pipeline, pairing an Artificial Neural Network (ANN) core with a real-time operational dashboard for immediate stakeholder decision-making.

---

## 📑 Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Architecture & Data Flow](#architecture--data-flow)

---

## 🔍 Overview

### Problem Statement
Customer acquisition costs (CAC) in the retail banking sector are substantially higher than customer retention costs. Unidentified structural patterns in account usage, demographic transitions, and financial balances often lead to silent customer attrition ("churn"). Identifying these high-risk customers before they close their accounts is a critical requirement for maintaining capital stability and optimizing retention marketing workflows.

### Solution
This project delivers a multi-layer deep learning classifier trained on complex historical customer profiles. It maps demographic vectors (Geography, Gender, Age) and financial indicators (Credit Score, Balance, Active Status, Product Density) to a highly calibrated, continuous probability score indicating churn risk.

### Target Users
- **Customer Relationship Managers (CRMs):** For automated risk flagging during customer touchpoints.
- **Business Intelligence & Analytics Teams:** For batch risk stratification and cohort tracking.
- **Product Owners:** For evaluating feature-level triggers (e.g., how balance drops correlate with exit velocity).

---

## ⚡ Features

- **Deterministic Preprocessing Architecture:** Complete preservation of training-state parameters through serialized `StandardScaler` and categorical encoders (`LabelEncoder`, `OneHotEncoder`), establishing zero data leakage.
- **Optimized Feed-Forward Architecture:** Multi-layer perceptron (MLP) built with TensorFlow/Keras using custom weight initializations, non-linear hidden activations (`ReLU`), and a final probabilistic layer (`Sigmoid`).
- **Interactive Operational UI:** A high-performance dashboard implemented via Streamlit, allowing ad-hoc real-time single-customer inferencing.
- **Early-Stopping & Convergence Safeguards:** Automated training termination utilizing validation loss monitors to prevent overfitting during long epochs.
- **Granular Feature Tracking:** Structural support for logging model parameters and historical performance characteristics via TensorBoard.

---

## 🏗️ Architecture & Data Flow

The platform relies on a decoupled, pipeline-centric design. Preprocessing state objects are treated as code dependencies during inference, guaranteeing mathematical consistency across training and production environments.

```mermaid
graph TD
    A[Raw UI Inputs / CSV Input] --> B[Feature Isolation Engine]
    B --> C[Categorical Path]
    B --> D[Numerical Path]
    
    C --> C1[LabelEncoder: Gender]
    C --> C2[OneHotEncoder: Geography]
    D --> D1[StandardScaler: CreditScore, Age, Balance, etc.]
    
    C1 --> E[Feature Vector Alignment & Alignment Matrix]
    C2 --> E
    D1 --> E
    
    E --> F[Serialized ANN Core]
    F --> F1[Dense Hidden Layer 1: 64 Neurons + ReLU]
    F1 --> F2[Dense Hidden Layer 2: 32 Neurons + ReLU]
    F2 --> F3[Output Layer: 1 Neuron + Sigmoid]
    
    F3 --> G[Probabilistic Risk Score 0.0 - 1.0]
    G --> H[Inference Boundary Engine: Churn / Retain]
