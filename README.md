# README: Reproducing MNIST Classification Experiments

## 📌 **Overview**

This project implements and evaluates three supervised learning algorithms—**Logistic Regression, Decision Tree, and Random Forest**—on the MNIST dataset of handwritten digits. The goal is to compare model performance based on accuracy, precision, recall, and F1-score.

## 🖥 **Requirements**

Before running the code, install the necessary dependencies. You can use the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

Alternatively, if using **Google Colab**, the required libraries are already pre-installed.

## 📂 **Dataset**

This project uses the **MNIST dataset**, which consists of **60,000 training** and **10,000 test images** of handwritten digits (0-9). The dataset is automatically downloaded when running the provided script.

## 🚀 **Steps to Reproduce the Experiments**

1. **Clone the repository (if applicable):**

   ```bash
   git clone https://github.com/AkiAkzh/methods_and_tools.git
   cd https://github.com/AkiAkzh/methods_and_tools.git
   ```

2. **Open the Jupyter Notebook (**\`\`\*\* file)\*\* in Google Colab or Jupyter Notebook.

3. **Run the following steps in order:**

   - **Load and preprocess the dataset**: Convert images to a 1D array, normalize pixel values, and split data into train/validation/test sets.
   - **Perform Exploratory Data Analysis (EDA)**: Visualize digit distribution and correlation matrix.
   - **Train three classification models**:
     - Logistic Regression
     - Decision Tree
     - Random Forest
   - **Evaluate model performance** using classification metrics (Accuracy, Precision, Recall, F1-score, and Confusion Matrix).
   - **Plot ROC curves** to compare model performance.
   - **Analyze misclassified examples** for further insights.

4. **Modify hyperparameters (Optional)**:

   - Tune `max_depth`, `min_samples_split` for Decision Tree and Random Forest.
   - Adjust `max_iter` for Logistic Regression to improve convergence.

5. **Run all cells** and compare the final results.

## 📊 **Expected Results**

After executing the notebook, you should see:

- **Logistic Regression Accuracy:** \~92%
- **Decision Tree Accuracy:** \~88%
- **Random Forest Accuracy:** \~97%
- **Confusion Matrices** for all models.
- **ROC Curves and AUC Scores** comparing classification performance.
- **Examples of misclassified digits**.

## ⚡ **Additional Notes**

- The notebook is optimized for Google Colab but can also run locally.
- Feature engineering or PCA can be added for further optimization.
- Consider using **Convolutional Neural Networks (CNNs)** for more advanced classification.

## 🔗 **References**

- MNIST Dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- Scikit-learn Documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- TensorFlow Keras Documentation: [https://www.tensorflow.org/api\_docs](https://www.tensorflow.org/api_docs)

---

📌 **Now you are ready to run the experiments and analyze the performance of different classification models!** 🚀

