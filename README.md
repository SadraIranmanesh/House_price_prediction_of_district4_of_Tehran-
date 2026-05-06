<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>House Price Prediction with PyTorch</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      line-height: 1.7;
      max-width: 1000px;
      margin: 40px auto;
      padding: 0 20px;
      color: #222;
      background: #fff;
    }
    h1, h2, h3 {
      color: #111;
    }
    code, pre {
      background: #f4f4f4;
      border-radius: 6px;
      font-family: Consolas, monospace;
    }
    code {
      padding: 2px 6px;
    }
    pre {
      padding: 14px;
      overflow-x: auto;
    }
    ul, ol {
      margin-bottom: 20px;
    }
  </style>
</head>
<body>

  <h1>House Price Prediction with PyTorch</h1>

  <p>
    A complete machine learning pipeline for <strong>house price prediction</strong> using <strong>PyTorch</strong>, including:
  </p>

  <ul>
    <li>data loading</li>
    <li>missing value handling</li>
    <li>outlier removal</li>
    <li>exploratory data analysis (EDA)</li>
    <li>feature scaling</li>
    <li>neural network regression</li>
    <li>early stopping</li>
    <li>model evaluation</li>
    <li>residual analysis</li>
    <li>visualization</li>
  </ul>

  <h2>Overview</h2>

  <p>
    This project predicts <strong>house prices</strong> using a fully connected neural network built with <strong>PyTorch</strong>.
  </p>

  <p>The workflow covers the full ML pipeline:</p>

  <ol>
    <li>Load and inspect dataset</li>
    <li>Clean data by removing missing-value columns</li>
    <li>Remove price outliers using the <strong>IQR method</strong></li>
    <li>Perform exploratory data analysis</li>
    <li>Split data into train / validation / test sets</li>
    <li>Scale features and target values using <strong>MinMaxScaler</strong></li>
    <li>Train a neural network regressor</li>
    <li>Apply <strong>early stopping</strong> to prevent overfitting</li>
    <li>
      Evaluate the model using:
      <ul>
        <li><strong>R² Score</strong></li>
        <li><strong>MAE</strong></li>
        <li><strong>RMSE</strong></li>
      </ul>
    </li>
    <li>Visualize predictions and residuals</li>
  </ol>

  <h2>Tech Stack</h2>

  <ul>
    <li><strong>Python</strong></li>
    <li><strong>Pandas</strong></li>
    <li><strong>NumPy</strong></li>
    <li><strong>Matplotlib</strong></li>
    <li><strong>Seaborn</strong></li>
    <li><strong>Scikit-learn</strong></li>
    <li><strong>PyTorch</strong></li>
  </ul>

  <h2>Project Structure</h2>

  <pre><code>├── README.md
├── best_model.pth
├── main.py
└── HomeDataset_after_preprocess_04_06.csv</code></pre>

  <p>You can rename <code>main.py</code> depending on your actual file name.</p>

  <h2>Features</h2>

  <ul>
    <li>End-to-end regression pipeline</li>
    <li>Outlier removal with IQR</li>
    <li>Correlation analysis and feature importance</li>
    <li>Neural network regression with PyTorch</li>
    <li>GPU support (<code>cuda</code> if available)</li>
    <li>Early stopping with best model checkpoint saving</li>
    <li>Training/validation loss tracking</li>
    <li>Prediction vs actual visualization</li>
    <li>Residual diagnostics</li>
  </ul>

  <h2>Dataset</h2>

  <p>
    The model expects a CSV dataset containing housing features and a target column named:
  </p>

  <pre><code>price</code></pre>

  <p>Example dataset loading path in the code:</p>

  <pre><code>df = pd.read_csv('HomeDataset_after_preprocess_04_06.csv')</code></pre>

  <h3>Important</h3>

  <p>
    Make sure the dataset file is placed in the project root directory, or update the path if needed.
  </p>

  <h2>Data Preprocessing</h2>

  <h3>1. Missing Values</h3>

  <p>Columns containing missing values are removed:</p>

  <pre><code>df = df.dropna(axis=1)</code></pre>

  <h3>2. Outlier Removal</h3>

  <p>
    Outliers are removed from the <code>price</code> column using the <strong>Interquartile Range (IQR)</strong> method:
  </p>

  <ul>
    <li>Lower bound = Q1 - 1.5 × IQR</li>
    <li>Upper bound = Q3 + 1.5 × IQR</li>
  </ul>

  <p>
    This helps reduce the effect of extreme house prices on model performance.
  </p>

  <h3>3. Feature / Target Split</h3>

  <ul>
    <li><strong>Features:</strong> all columns except <code>price</code></li>
    <li><strong>Target:</strong> <code>price</code></li>
  </ul>

  <h3>4. Train / Validation / Test Split</h3>

  <ul>
    <li><strong>70%</strong> Training</li>
    <li><strong>15%</strong> Validation</li>
    <li><strong>15%</strong> Test</li>
  </ul>

  <h3>5. Scaling</h3>

  <p>Both features and target are normalized using <strong>MinMaxScaler</strong>.</p>

  <h2>Exploratory Data Analysis</h2>

  <p>The project includes several visualizations for understanding the dataset:</p>

  <ul>
    <li>correlation heatmap</li>
    <li>price distribution histogram</li>
    <li>price boxplot</li>
    <li>feature correlation with price</li>
    <li>training vs validation loss</li>
    <li>actual vs predicted prices</li>
    <li>residual scatter plot</li>
    <li>residual distribution</li>
  </ul>

  <h2>Model Architecture</h2>

  <p>The model is a feed-forward neural network implemented in PyTorch.</p>

  <h3>Architecture</h3>

  <pre><code>Input Layer
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
Linear(8 → 1)</code></pre>

  <h3>Loss Function</h3>

  <ul>
    <li><strong>Mean Squared Error (MSE)</strong></li>
  </ul>

  <h3>Optimizer</h3>

  <ul>
    <li><strong>Adam</strong></li>
    <li>Learning rate: <code>0.01</code></li>
    <li>Weight decay: <code>1e-4</code></li>
  </ul>

  <h2>Training Configuration</h2>

  <ul>
    <li><strong>Epochs:</strong> 10000</li>
    <li><strong>Batch size:</strong> 16</li>
    <li><strong>Early stopping patience:</strong> 500</li>
    <li><strong>Random seed:</strong> 42</li>
  </ul>

  <p>The best validation model is saved automatically as:</p>

  <pre><code>best_model.pth</code></pre>

  <h2>Evaluation Metrics</h2>

  <p>After training, the model is evaluated on the test set using:</p>

  <ul>
    <li><strong>R² Score</strong></li>
    <li><strong>Mean Absolute Error (MAE)</strong></li>
    <li><strong>Root Mean Squared Error (RMSE)</strong></li>
  </ul>

  <p>Example:</p>

  <pre><code>r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))</code></pre>

  <h2>Installation</h2>

  <p>Clone the repository:</p>

  <pre><code>git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name</code></pre>

  <p>Install dependencies:</p>

  <pre><code>pip install pandas numpy matplotlib seaborn scikit-learn torch</code></pre>

  <h2>Usage</h2>

  <p>Run the training script:</p>

  <pre><code>python main.py</code></pre>

  <p>If you are using a notebook, simply run the cells step by step.</p>

  <h2>Output</h2>

  <p>During execution, the project will:</p>

  <ul>
    <li>print dataset information</li>
    <li>show missing values report</li>
    <li>remove outliers</li>
    <li>display EDA plots</li>
    <li>train the neural network</li>
    <li>save the best model</li>
    <li>print final evaluation metrics</li>
    <li>generate prediction and residual plots</li>
  </ul>

  <h2>Example Workflow</h2>

  <pre><code># Load data
df = pd.read_csv('HomeDataset_after_preprocess_04_06.csv')

# Preprocess
df = df.dropna(axis=1)

# Train model
# Evaluate model
# Visualize results</code></pre>

  <h2>Model Strengths</h2>

  <ul>
    <li>Simple and effective deep learning baseline for tabular regression</li>
    <li>Includes validation monitoring and early stopping</li>
    <li>Good structure for further experimentation and improvement</li>
    <li>Easy to adapt to other regression datasets</li>
  </ul>

  <h2>Possible Improvements</h2>

  <p>Here are some ideas to improve the project further:</p>

  <ul>
    <li>use <strong>DataLoader</strong> for cleaner batching</li>
    <li>add <strong>model checkpointing</strong> with full training state</li>
    <li>log metrics with <strong>TensorBoard</strong></li>
    <li>experiment with deeper/wider architectures</li>
    <li>apply <strong>feature selection</strong></li>
    <li>
      compare performance with:
      <ul>
        <li>Linear Regression</li>
        <li>Random Forest Regressor</li>
        <li>XGBoost</li>
        <li>CatBoost</li>
      </ul>
    </li>
    <li>use <strong>k-fold cross-validation</strong></li>
    <li>save scalers for inference deployment</li>
    <li>build a small <strong>Flask/FastAPI</strong> app for prediction</li>
  </ul>

  <h2>Notes</h2>

  <ul>
    <li>The current code drops all columns containing missing values. Depending on the dataset, imputing missing values may preserve more useful information.</li>
    <li>Outlier removal is only applied to the target column (<code>price</code>). You may also consider handling feature-level outliers.</li>
    <li>Since this is tabular data, traditional ML models may also perform competitively and are worth comparing.</li>
  </ul>

  <h2>Requirements</h2>

  <p>You can also create a <code>requirements.txt</code> file like this:</p>

  <pre><code>pandas
numpy
matplotlib
seaborn
scikit-learn
torch</code></pre>

  <h2>License</h2>

  <p>
    This project is open-source and available under the <strong>MIT License</strong>.
  </p>

  <h2>Author</h2>

  <p>
    Developed by <strong>Sadra</strong>
  </p>

  <p>
    If you found this project useful, feel free to star the repository.
  </p>

</body>
</html>
