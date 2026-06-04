# 🌾 Kissan AI — Crop & Fertilizer Advisor

An AI-powered web app that recommends the best crop and fertilizer 
for Pakistani farmers based on their soil and weather conditions.

Built by Muhammad Arsalan | CS Student, Peshawar 🇵🇰

## Live Demo
https://kissan.streamlit.app

## What it does
- Takes soil nutrients (N, P, K), pH, temperature, humidity, rainfall as input
- Predicts the best crop to grow (22 possible crops)
- Recommends the right fertilizer for that crop and soil type

## Models
- Crop Model: Random Forest — 99.32% accuracy
- Fertilizer Model: Random Forest — 95.0% accuracy

## Tech Stack
- Python, Scikit-learn, Pandas, Streamlit

## Run Locally
pip install -r requirements.txt
python train.py
streamlit run app.py