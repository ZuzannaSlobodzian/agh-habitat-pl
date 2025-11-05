# 🇵🇱 AGH Habitat-PL
*A starter repository for the course **Machine Learning for Space Applications** (AGH 2025/26)*

---

## 🎯 Project goal
Build an **AI pipeline to classify natural habitats across Poland** using **Google Satellite Embeddings** and open geospatial reference data (e.g., CORINE, Natura 2000).  
Each lab adds one methodological layer — from data access to deep learning and spatial validation.

---

## Lab 1: Setup, authentication, data access

*Google Colab Notebook:* https://colab.research.google.com/drive/1kdSxB3m3NTL4booz-w8FoCI7YFzpxDWU?usp=sharing

*Datasets*

1.   Google Satellite Embeddings
2.   CORINE 2018

*Requirements*

- Google account (provide Google email to instructor)
- Github account
- Github access token

*Setup and authentication steps*

- Clone Github repository
- Check out template repository into Google Colab
- Generate Github access token
- Setup GITHUB_TOKEN in Secrets
- Varify access to Google Earth Engine
- Visualize datasets

*Objectives*

1. Define two AOIs for well distinct, well defined surface classes
2. Download Google Sat Embeddings for the defined AOIs
3. Apply unsupervised clustering of the embeddings using 4 different clustering methods
4. Visualise and document the results
