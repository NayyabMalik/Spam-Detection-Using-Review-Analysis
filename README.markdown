# Spam Detection Using Review Analysis

This repository contains a spam detection project that analyzes reviews from the Women’s Clothing E-Commerce Reviews dataset to classify reviews as spam or non-spam based on sentiment and rating analysis. The project employs multiple machine learning and deep learning techniques, including Naive Bayes, Logistic Regression, Random Forest, Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM) networks, and BERT-based Transformer models. It includes graphical analyses to explore review characteristics and model performance.

The project is part of my coursework at the National University of Modern Languages, Islamabad, submitted on October 31, 2024, under the supervision of Mam Iqra Nasem. It builds on concepts from my deep learning labs, particularly neural network-based classification and text processing.

## Project Overview

The goal is to detect spam reviews in an e-commerce dataset by analyzing review text and ratings. Low-rated reviews (e.g., 1-2 stars) or those with negative sentiment are considered potential spam. The dataset contains 19,662 reviews with features like `Review Text`, `Rating`, `Age`, and `Recommended IND`. The project involves:

- **Data Preprocessing**: Cleaning text by removing non-alphabetic characters, converting to lowercase, and tokenizing.
- **Feature Extraction**: Using CountVectorizer, TfidfVectorizer, and Word2Vec to represent review text.
- **Classification Models**:
  - Naive Bayes (CountVectorizer: 61.9%, TfidfVectorizer: 58.7%)
  - Logistic Regression (CountVectorizer: 60.6%, Word2Vec: 62.3%)
  - Random Forest (57.6%)
  - RNN (49.1%)
  - LSTM (57.3%)
  - BERT-based Transformer for sentiment analysis
- **Graphical Analysis**: Visualizations including rating distribution, word cloud, review length by rating, and sentiment distribution.

## Dataset

The dataset (`Womens Clothing E-Commerce Reviews.csv`) contains 19,662 reviews with the following columns:

| Column                  | Description                                    | Dtype  |
|-------------------------|------------------------------------------------|--------|
| Unnamed: 0              | Index column                                   | int64  |
| Clothing ID             | Identifier for clothing item                   | int64  |
| Age                     | Reviewer’s age                                 | int64  |
| Title                   | Review title                                   | object |
| Review Text             | Full review text                               | object |
| Rating                  | Rating (1-5 stars)                             | int64  |
| Recommended IND         | Recommendation indicator (0 or 1)              | int64  |
| Positive Feedback Count | Number of positive feedbacks                   | int64  |
| Division Name           | Division (e.g., General, Initmates)            | object |
| Department Name         | Department (e.g., Dresses, Tops)               | object |
| Class Name              | Clothing category (e.g., Knits, Blouses)       | object |
| Review Length           | Derived: Length of review text (characters)    | int64  |

**Source**: [Kaggle Women’s Clothing E-Commerce Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

## Repository Structure

```
spam-detection-reviews/
├── data/
│   ├── Womens Clothing E-Commerce Reviews.csv  # Dataset
├── notebooks/
│   ├── spam_detection_analysis.ipynb           # Data preprocessing, visualization, and modeling
├── static/
│   ├── images/
│   │   ├── rating_distribution.png            # Bar chart of ratings
│   │   ├── word_cloud.png                     # Word cloud of review text
│   │   ├── review_length_by_rating.png        # Box plot of review length
│   │   ├── sentiment_distribution.png         # Bar chart of BERT sentiment
├── requirements.txt                               # Python dependencies
├── README.md                                     # This file
├── LICENSE                                       # MIT License
```

## Methodology

1. **Data Preprocessing**:
   - Removed rows with missing `Review Text` or `Rating`.
   - Cleaned text: Removed non-alphabetic characters, converted to lowercase, tokenized.
   - Created `Review Length` feature for analysis.

2. **Feature Extraction**:
   - **CountVectorizer**: Bag-of-words model with 5,000 features, excluding English stop words.
   - **TfidfVectorizer**: Term frequency-inverse document frequency with 5,000 features.
   - **Word2Vec**: Word embeddings with 100-dimensional vectors, averaging word vectors per review.

3. **Classification Models**:
   - **Naive Bayes**: MultinomialNB with CountVectorizer (61.9% accuracy) and TfidfVectorizer (58.7% accuracy).
   - **Logistic Regression**: Applied with CountVectorizer (60.6% accuracy) and Word2Vec (62.3% accuracy).
   - **Random Forest**: 100 estimators with CountVectorizer (57.6% accuracy).
   - **RNN**: Simple RNN with embedding layer (49.1% accuracy).
   - **LSTM**: LSTM with embedding layer (57.3% accuracy).
   - **BERT**: Pre-trained `bert-base-uncased` for sentiment analysis, mapping reviews to positive/negative labels.

4. **Graphical Analysis**:
   - **Rating Distribution**: Bar chart showing the frequency of 1-5 star ratings.
   - **Word Cloud**: Visualizes common words in reviews.
   - **Review Length by Rating**: Box plot showing review length distribution across ratings.
   - **Sentiment Distribution**: Bar chart of BERT-based positive/negative sentiment counts.

## Results

| Model                  | Feature Extraction | Accuracy (%) | Notes                                                                 |
|------------------------|--------------------|--------------|----------------------------------------------------------------------|
| Naive Bayes            | CountVectorizer    | 61.9         | Strong performance on high ratings (5), weaker on low ratings (1-2). |
| Naive Bayes            | TfidfVectorizer    | 58.7         | Lower recall for low ratings, biased toward rating 5.                 |
| Logistic Regression    | CountVectorizer    | 60.6         | Balanced performance across ratings.                                 |
| Logistic Regression    | Word2Vec           | 62.3         | Best overall accuracy, improved low-rating detection.                 |
| Random Forest          | CountVectorizer    | 57.6         | High bias toward rating 5, poor low-rating recall.                    |
| RNN                    | Tokenized Sequences | 49.1         | Overfitting observed, low validation accuracy.                       |
| LSTM                   | Tokenized Sequences | 57.3         | Better than RNN but still struggles with low ratings.                 |
| BERT (Sentiment)       | Transformer        | N/A          | Used for binary sentiment (positive/negative), not rating prediction. |

**Key Observations**:
- Logistic Regression with Word2Vec achieved the highest accuracy (62.3%).
- Models struggle with low-rated reviews (1-2 stars) due to class imbalance (57% of reviews are rated 5).
- BERT provides qualitative sentiment insights but is not directly used for rating prediction.

## Related Coursework

This project builds on my deep learning labs, particularly:

- **Lab 1: ANN_Classification** (`deep-learning-labs/lab_manuals/ANN_Classification.pdf`): Neural network fundamentals, relevant to RNN and LSTM models.
- **Lab 11: RNN_Text** (`deep-learning-labs/lab_manuals/RNN_Text.pdf`): Sequence modeling for text, applied to RNN/LSTM for review analysis.
- **Lab 8: Speech Signal Classification** (`speech-processing-labs/lab_reports/Lab8_Speech_Classification.pdf`): Neural network-based classification, similar to review classification.

See the `deep-learning-labs` and `speech-processing-labs` repositories for details.

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/spam-detection-reviews.git
   cd spam-detection-reviews
   ```

2. **Install Dependencies**:

   Install Python libraries listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   Key libraries: `pandas`, `matplotlib`, `seaborn`, `wordcloud`, `scikit-learn`, `tensorflow`, `transformers`, `gensim`, `numpy`.

3. **Download Dataset**:

   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews).
   - Place it in `data/Womens Clothing E-Commerce Reviews.csv`.

4. **Run the Notebook**:

   Launch Jupyter Notebook and execute the analysis:

   ```bash
   jupyter notebook notebooks/spam_detection_analysis.ipynb
   ```

5. **View Visualizations**:

   Visualizations are saved in `static/images/` (e.g., `rating_distribution.png`, `word_cloud.png`).

## Usage

1. **Data Preprocessing**:
   - Load the dataset and clean `Review Text` by removing non-alphabetic characters and converting to lowercase.
   - Compute `Review Length` for analysis.

2. **Feature Extraction**:
   - Use `CountVectorizer` or `TfidfVectorizer` for bag-of-words or TF-IDF features.
   - Train Word2Vec for word embeddings.
   - Tokenize and pad sequences for RNN/LSTM models.

3. **Model Training and Evaluation**:
   - Train classifiers (Naive Bayes, Logistic Regression, Random Forest) and evaluate using accuracy and classification reports.
   - Train RNN and LSTM models with tokenized sequences.
   - Apply BERT for sentiment analysis.

4. **Graphical Analysis**:
   - Generate visualizations to explore data and model performance (saved in `static/images/`).

**Example** (Logistic Regression with Word2Vec):

```python
import pandas as pd
import re
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/Womens Clothing E-Commerce Reviews.csv")
message = [[word for word in re.sub("[^A-Za-z\s]", ' ', str(text)).lower().split()] for text in df['Review Text']]
model = Word2Vec(sentences=message, vector_size=100, window=5, min_count=5, sg=0)
X = np.array([np.mean([model.wv[word] for word in review if word in model.wv] or [np.zeros(100)], axis=0) for review in message])
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

## Future Improvements

- **Class Imbalance**: Apply oversampling (e.g., SMOTE) or weighted loss functions to improve low-rating (1-2) detection.
- **Model Enhancement**: Fine-tune BERT for multi-class rating prediction instead of binary sentiment.
- **Feature Engineering**: Incorporate additional features like `Age`, `Positive Feedback Count`, or `Recommended IND`.
- **Web Interface**: Develop a Flask-based interface (similar to `sales-forecasting/app.py`) for interactive review analysis.
- **Evaluation Metrics**: Add ROC curves and confusion matrices for deeper model insights.

## Notes

- **File Size**: Use Git LFS for the dataset and visualizations (`git lfs track "*.csv" "*.png"`).
- **Permissions**: The code and summarized coursework are shared with permission for educational purposes. Contact Mam Iqra Nasem for original materials.
- **Class Imbalance**: The dataset is skewed toward 5-star ratings (57%), impacting model performance on low ratings.
- **BERT Limitation**: Current BERT implementation is for binary sentiment; multi-class adaptation requires fine-tuning.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.