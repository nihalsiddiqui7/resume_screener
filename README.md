# AI-Powered Resume Screening & Classification

![Resume Screening App Demo](https://user-images.githubusercontent.com/your-username/your-repo/assets/demo.gif)


**[Live Demo](https://nihalsiddiqui7-resume-screener-notebookapp-f4ixv4.streamlit.app/)** 🚀

---

## 🎯 Project Overview

In today's competitive job market, recruiters review hundreds of resumes for a single position. This manual process is time-consuming, inefficient, and prone to bias. This project introduces an intelligent **Resume Screening Application** that automates the initial screening phase by leveraging Natural Language Processing (NLP) and Machine Learning.

The application parses resumes in various formats (PDF, DOCX), cleans the text data, and classifies them into predefined job categories using a trained classification model. This allows recruiters to quickly identify the most suitable candidates, significantly reducing hiring time and improving efficiency.

---

## ✨ Key Features

-   **Automated Classification**: Automatically categorizes resumes into roles like 'Data Science', 'Web Development', 'HR', etc.
-   **Multi-Format Support**: Seamlessly extracts text from both `.pdf` and `.docx` file formats.
-   **NLP-Powered Cleaning**: Implements a robust text preprocessing pipeline to clean and normalize resume data for accurate predictions.
-   **Interactive UI**: A clean, intuitive, and user-friendly web interface built with Streamlit.
-   **Scalable & Deployable**: The application is container-ready and designed for easy deployment on cloud platforms like Streamlit Cloud.

---

## 🛠️ Tech Stack

| Area                  | Technologies & Libraries                                     |
| --------------------- | ------------------------------------------------------------ |
| **Backend & ML**      | Python, Scikit-learn, NLTK, Pandas, NumPy                    |
| **Web Framework**     | Streamlit                                                    |
| **File Handling**     | PyPDF2, python-docx                                          |
| **Version Control**   | Git & GitHub                                                 |
| **Deployment**        | Streamlit Cloud                                              |

---
### Model Performance

The final model was evaluated on a held-out test set (20% of the data).

| Metric              | Score       |
| ------------------- | ----------- |
| **Accuracy**        | **`98.44%`** |
| **Precision (Macro)** | `0.98`      |
| **Recall (Macro)**    | `0.98`      |
| **F1-Score (Macro)**  | `0.98`      |

---

## ⚙️ System Workflow

The application follows a simple yet powerful workflow:

1.  **Upload**: The user uploads a resume file via the web interface.
2.  **Parse**: The system detects the file type and extracts raw text.
3.  **Preprocess**: The text is cleaned by removing stopwords, punctuation, and special characters.
4.  **Transform**: The cleaned text is converted into a numerical vector using a pre-trained TF-IDF Vectorizer.
5.  **Predict**: The machine learning model (KNN with OVR) predicts the job category based on the vector.
6.  **Display**: The predicted category is displayed to the user.

---

## 🚀 Getting Started

You can run this application on your local machine by following these steps.

### Prerequisites

-   Python 3.8+
-   Git

### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nihalsiddiqui7/resume_screener.git
    cd resume_screener
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run notebook/app.py
    ```

5.  Open your web browser and navigate to `http://localhost:8501`.

---

## 📈 Model Details

-   **Algorithm**: A **One-vs-Rest (OVR) Classifier** strategy was used, allowing a binary classifier (KNeighbors Classifier) to handle a multi-class classification problem. This approach is efficient and provides good baseline performance.
-   **Feature Extraction**: **TF-IDF (Term Frequency-Inverse Document Frequency)** was employed to convert the text data into meaningful numerical features, capturing the importance of words relative to the collection of all resumes.

---

## 📬 Contact

I'm passionate about leveraging data to build impactful solutions. Let's connect!

-   **LinkedIn**: [https://www.linkedin.com/in/nihal-siddiqui-49593b268/](https://www.linkedin.com/in/nihal-siddiqui-49593b268/)
-   **Email**: `nihal070502@gmail.com`
-   

---
*Give this project a ⭐ if you found it interesting!*
