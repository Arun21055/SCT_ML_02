# 🛍️ Customer Segmentation Using K-Means Clustering

This project applies **K-Means Clustering** to group customers based on their annual income and spending behavior. The dataset is taken from a mall's customer data and is a classic example used in unsupervised machine learning tasks.

---

## 📁 Dataset

The dataset `Mall_Customers.csv` contains information about customers including:
- `CustomerID`: Unique customer ID
- `Gender`: Male/Female
- `Age`: Age of the customer
- `Annual Income (k$)`: Annual income in thousands
- `Spending Score (1-100)`: Score assigned based on spending patterns

---

## 🧠 Objective

Segment customers into **distinct groups** based on:
- Annual Income
- Spending Score

These segments can be used for:
- Targeted marketing
- Personalized promotions
- Customer relationship management

---

## 🛠️ Technologies Used

- Python 🐍
- Pandas for data manipulation
- Scikit-learn for clustering
- Matplotlib for visualization

---

## 🚀 Steps Performed

1. **Data Preprocessing**
   - Cleaned and stripped column names
   - Standardized features using `StandardScaler`

2. **Elbow Method**
   - Plotted WCSS to find optimal `K` (number of clusters)

3. **K-Means Clustering**
   - Applied clustering with optimal K (K=5)
   - Visualized customer segments

4. **Statistical Analysis**
   - Printed cluster-wise average Age, Income, and Spending Score
   - Printed count of customers per cluster

---

## 📊 Results

- Customers were successfully grouped into 5 clusters
- Each cluster shows distinct behavior (e.g., high income but low spending, or low income with high spending)
- The results can guide targeted business strategies

---

## 📌 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/customer-segmentation-kmeans.git
   cd customer-segmentation-kmeans
