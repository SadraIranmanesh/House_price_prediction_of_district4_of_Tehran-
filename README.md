\section{House Price Prediction with PyTorch}

A complete machine learning pipeline for \textbf{house price prediction} using \textbf{PyTorch}, including:

\begin{itemize}
    \item data loading
    \item missing value handling
    \item outlier removal
    \item exploratory data analysis (EDA)
    \item feature scaling
    \item neural network regression
    \item early stopping
    \item model evaluation
    \item residual analysis
    \item visualization
\end{itemize}

\section{Overview}

This project predicts \textbf{house prices} using a fully connected neural network built with \textbf{PyTorch}.

The workflow covers the full ML pipeline:

\begin{enumerate}
    \item Load and inspect dataset
    \item Clean data by removing missing-value columns
    \item Remove price outliers using the \textbf{IQR method}
    \item Perform exploratory data analysis
    \item Split data into train / validation / test sets
    \item Scale features and target values using \textbf{MinMaxScaler}
    \item Train a neural network regressor
    \item Apply \textbf{early stopping} to prevent overfitting
    \item Evaluate the model using:
    \begin{itemize}
        \item \textbf{R\textsuperscript{2} Score}
        \item \textbf{MAE}
        \item \textbf{RMSE}
    \end{itemize}
    \item Visualize predictions and residuals
\end{enumerate}

\section{Tech Stack}

\begin{itemize}
    \item \textbf{Python}
    \item \textbf{Pandas}
    \item \textbf{NumPy}
    \item \textbf{Matplotlib}
    \item \textbf{Seaborn}
    \item \textbf{Scikit-learn}
    \item \textbf{PyTorch}
\end{itemize}

\section{Project Structure}

\begin{verbatim}
├── README.md
├── best_model.pth
├── main.py
└── HomeDataset_after_preprocess_04_06.csv
\end{verbatim}

You can rename \texttt{main.py} depending on your actual file name.

\section{Features}

\begin{itemize}
    \item End-to-end regression pipeline
    \item Outlier removal with IQR
    \item Correlation analysis and feature importance
    \item Neural network regression with PyTorch
    \item GPU support (\texttt{cuda} if available)
    \item Early stopping with best model checkpoint saving
    \item Training/validation loss tracking
    \item Prediction vs actual visualization
    \item Residual diagnostics
\end{itemize}

\section{Dataset}

The model expects a CSV dataset containing housing features and a target column named:

\begin{verbatim}
price
\end{verbatim}

Example dataset loading path in the code:

\begin{verbatim}
df = pd.read_csv('HomeDataset_after_preprocess_04_06.csv')
\end{verbatim}

\subsection*{Important}

Make sure the dataset file is placed in the project root directory, or update the path if needed.

\section{Data Preprocessing}

\subsection{1. Missing Values}

Columns containing missing values are removed:

\begin{verbatim}
df = df.dropna(axis=1)
\end{verbatim}

\subsection{2. Outlier Removal}

Outliers are removed from the \texttt{price} column using the \textbf{Interquartile Range (IQR)} method:

\begin{itemize}
    \item Lower bound = $Q1 - 1.5 \times IQR$
    \item Upper bound = $Q3 + 1.5 \times IQR$
\end{itemize}

This helps reduce the effect of extreme house prices on model performance.

\subsection{3. Feature / Target Split}

\begin{itemize}
    \item \textbf{Features:} all columns except \texttt{price}
    \item \textbf{Target:} \texttt{price}
\end{itemize}

\subsection{4. Train / Validation / Test Split}

\begin{itemize}
    \item \textbf{70\%} Training
    \item \textbf{15\%} Validation
    \item \textbf{15\%} Test
\end{itemize}

\subsection{5. Scaling}

Both features and target are normalized using \textbf{MinMaxScaler}.

\section{Exploratory Data Analysis}

The project includes several visualizations for understanding the dataset:

\begin{itemize}
    \item correlation heatmap
    \item price distribution histogram
    \item price boxplot
    \item feature correlation with price
    \item training vs validation loss
    \item actual vs predicted prices
    \item residual scatter plot
    \item residual distribution
\end{itemize}

\section{Model Architecture}

The model is a feed-forward neural network implemented in PyTorch.

\subsection*{Architecture}

\begin{verbatim}
Input Layer
   ↓
Linear(input_size → 64)
ReLU
Dropout(0.5)
   ↓
Linear(64 → 16)
ReLU
   ↓
Linear(16 → 8)
ReLU
   ↓
Linear(8 → 1)
\end{verbatim}

\subsection*{Loss Function}

\begin{itemize}
    \item \textbf{Mean Squared Error (MSE)}
\end{itemize}

\subsection*{Optimizer}

\begin{itemize}
    \item \textbf{Adam}
    \item Learning rate: \texttt{0.01}
    \item Weight decay: \texttt{1e-4}
\end{itemize}

\section{Training Configuration}

\begin{itemize}
    \item \textbf{Epochs:} 10000
    \item \textbf{Batch size:} 16
    \item \textbf{Early stopping patience:} 500
    \item \textbf{Random seed:} 42
\end{itemize}

The best validation model is saved automatically as:

\begin{verbatim}
best_model.pth
\end{verbatim}

\section{Evaluation Metrics}

After training, the model is evaluated on the test set using:

\begin{itemize}
    \item \textbf{R\textsuperscript{2} Score}
    \item \textbf{Mean Absolute Error (MAE)}
    \item \textbf{Root Mean Squared Error (RMSE)}
\end{itemize}

Example:

\begin{verbatim}
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
\end{verbatim}

\section{Installation}

Clone the repository:

\begin{verbatim}
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
\end{verbatim}

Install dependencies:

\begin{verbatim}
pip install pandas numpy matplotlib seaborn scikit-learn torch
\end{verbatim}

\section{Usage}

Run the training script:

\begin{verbatim}
python main.py
\end{verbatim}

If you are using a notebook, simply run the cells step by step.

\section{Output}

During execution, the project will:

\begin{itemize}
    \item print dataset information
    \item show missing values report
    \item remove outliers
    \item display EDA plots
    \item train the neural network
    \item save the best model
    \item print final evaluation metrics
    \item generate prediction and residual plots
\end{itemize}

\section{Example Workflow}

\begin{verbatim}
# Load data
df = pd.read_csv('HomeDataset_after_preprocess_04_06.csv')

# Preprocess
df = df.dropna(axis=1)

# Train model
# Evaluate model
# Visualize results
\end{verbatim}

\section{Model Strengths}

\begin{itemize}
    \item Simple and effective deep learning baseline for tabular regression
    \item Includes validation monitoring and early stopping
    \item Good structure for further experimentation and improvement
    \item Easy to adapt to other regression datasets
\end{itemize}

\section{Possible Improvements}

Here are some ideas to improve the project further:

\begin{itemize}
    \item use \textbf{DataLoader} for cleaner batching
    \item add \textbf{model checkpointing} with full training state
    \item log metrics with \textbf{TensorBoard}
    \item experiment with deeper/wider architectures
    \item apply \textbf{feature selection}
    \item compare performance with:
    \begin{itemize}
        \item Linear Regression
        \item Random Forest Regressor
        \item XGBoost
        \item CatBoost
    \end{itemize}
    \item use \textbf{k-fold cross-validation}
    \item save scalers for inference deployment
    \item build a small \textbf{Flask/FastAPI} app for prediction
\end{itemize}

\section{Notes}

\begin{itemize}
    \item The current code drops all columns containing missing values. Depending on the dataset, imputing missing values may preserve more useful information.
    \item Outlier removal is only applied to the target column (\texttt{price}). You may also consider handling feature-level outliers.
    \item Since this is tabular data, traditional ML models may also perform competitively and are worth comparing.
\end{itemize}

\section{Requirements}

You can also create a \texttt{requirements.txt} file like this:

\begin{verbatim}
pandas
numpy
matplotlib
seaborn
scikit-learn
torch
\end{verbatim}

\section{License}

This project is open-source and available under the \textbf{MIT License}.

\section{Author}

Developed by \textbf{Sadra}

If you found this project useful, feel free to star the repository.
