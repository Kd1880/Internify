# 🌟 Internify – AI-Powered Resume → Job Matching System

Internify is an AI/ML-powered system that analyzes resumes, extracts skills, and recommends the most relevant job roles using machine learning techniques such as TF-IDF, Logistic Regression, and KMeans clustering.

This repository contains:

- 🧠 Flask Backend API  
- 🎨 Streamlit Frontend  
- 🛢 SQLite Database Support  
- 📄 PDF Parsing + NLP  
- 📊 Job Matching Analytics  

---

## 🚀 Features

### 🔍 AI Resume Understanding
- PDF resume parsing  
- Text extraction using NLP  
- Skill & keyword detection  
- TF-IDF vectorization  

### 🤖 Machine Learning Pipeline
- Resume vectorization  
- Logistic Regression scoring  
- KMeans clustering  
- Weighted final score  
- Top job recommendations  

### 🧩 Backend API (Flask)
- `GET /` – API health check  
- `POST /signup`  
- `POST /login`  
- `POST /upload_resume`  
- `GET /matches?user_id=`  

### 🖥️ Frontend (Streamlit)
- Resume upload UI  
- Job recommendations  
- Score visualization (Pie chart + Histogram)  
- User login + history  

---

## 📂 Project Structure

Internify/
│
├── backend/
│ ├── app.py
│ ├── recommender_pipeline.py
│ ├── db_handler.py
│ ├── models/
│ │ ├── content_filter.py
│ │ ├── kmeans_model.py
│ │ ├── logistic_regression.py
│ │ └── nlp_parser.py
│ ├── utils/
│ │ ├── pdf_to_text.py
│ │ └── resume_parser.py
│ ├── uploads/ # (empty → contains .gitkeep)
│ └── database/ # (empty → contains .gitkeep)
│
├── frontend/
│ ├── streamlit_app.py
│ └── requirements.txt
│
├── requirements.txt
└── README.md


---

## 🔧 Installation & Setup

Clone the repo:

```bash
git clone https://github.com/<your-username>/internify.git
cd internify

▶️ Run Backend (Flask)

Install backend dependencies:

pip install -r requirements.txt


Run server:

cd backend
python app.py


API available at:

http://127.0.0.1:5000

🖥️ Run Frontend (Streamlit)

Open new terminal:

cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py


Open in browser:

http://localhost:8501

🧪 API Testing
Health Check
GET /

Signup
POST /signup
{
  "name": "test",
  "email": "test@example.com",
  "password": "pass123"
}

Upload Resume (example)
curl -X POST \
  -F "user_id=1" \
  -F "file=@resume.pdf" \
  http://127.0.0.1:5000/upload_resume

🛠 Tech Stack
Backend

Python

Flask

SQLite

scikit-learn

PyMuPDF

Frontend

Streamlit

Pandas

Matplotlib

ML/NLP

TF-IDF

Logistic Regression

KMeans

👨‍💻 Author

Kriti Dogra
AI/ML Developer & Designer
GitHub: https://github.com/Kd1880

📜 License

This project is licensed under the MIT License.
