from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

# Ensure the folder exists
os.makedirs("data/resumes", exist_ok=True)
w

def create_sample_resume(file_path):
    resume_text = """NIHAL SIDDIQUI
Data Science Enthusiast | Machine Learning | Python | SQL

Email: nihal.siddiqui@gmail.com
Phone: +91 9876543210
LinkedIn: linkedin.com/in/nihalsiddiqui

Summary:
Aspiring data scientist passionate about using data to solve real-world problems. Skilled in data analysis, visualization, and predictive modeling.

Education:
B.Tech in Computer Engineering - MGM University, 2022–2026

Skills:
Python, Pandas, NumPy, Scikit-learn, SQL, Data Visualization, Machine Learning, Deep Learning

Projects:
- House Price Prediction: Built a regression model using scikit-learn and deployed with Streamlit.
- Movie Recommendation System: Implemented collaborative filtering and cosine similarity.

Experience:
Data Science Intern at XYZ Analytics (Jun 2024 – Sep 2024)
- Performed data cleaning and visualization for customer churn analysis.
- Improved model accuracy by 8% using feature engineering.

Certifications:
Google Data Analytics Certificate
"""

    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter
    text_object = c.beginText(50, height - 50)
    text_object.setFont("Helvetica", 10)

    for line in resume_text.split("\n"):
        text_object.textLine(line)
    
    c.drawText(text_object)
    c.save()
    print(f"✅ Sample resume saved at: {file_path}")

# Run this
create_sample_resume("data/resumes/sample_resume.pdf")
