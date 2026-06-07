## Hate Speech Detection

This project uses the **Twitter Hate Speech and Offensive Language Dataset**, a publicly available dataset developed for research in text classification and hate speech detection. The dataset contains approximately **24,783 tweets**, each labeled into one of three categories:

* **0 – Hate Speech**
* **1 – Offensive Language**
* **2 – Neither Hate Speech nor Offensive Language**

The dataset (`twitter.csv`) is loaded and processed using the **Pandas** library. Text preprocessing is performed with **NLTK**, where tweets are converted to lowercase, and URLs, punctuation, numbers, and stopwords are removed. Stemming is applied to reduce words to their root form, improving the consistency of the data.

The cleaned text is then transformed into numerical features using **CountVectorizer** from **Scikit-learn**. A **Decision Tree Classifier** is trained on these features to learn patterns from the dataset and classify new user-provided text. The model's performance is evaluated using **Accuracy Score** and a **Confusion Matrix**, with visualization provided through **Matplotlib** and **Seaborn**.

### Libraries Used

* Pandas
* NumPy
* NLTK
* Scikit-learn
* Matplotlib
* Seaborn

### Outcome

The system allows users to enter a sentence and predicts whether it belongs to the categories of **Hate Speech**, **Offensive Language**, or **Neither Hate Speech nor Offensive Language**. This demonstrates the practical application of Natural Language Processing and Machine Learning for automated text moderation.

### Future Scope

* Improve prediction accuracy by using advanced models such as Logistic Regression, Random Forest, or Deep Learning techniques.
* Develop a web-based interface using Flask or Django.
* Integrate the model with social media platforms for real-time content moderation.
* Train the model on larger and more diverse datasets to improve its generalization capabilities.

