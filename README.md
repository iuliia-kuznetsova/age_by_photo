# Prediction of Final Steel Temperature at a Steel Production Plant

A machine learning model for predicting final molten steel temperature during ladle treatment in order to optimize energy consumption in steelmaking.


## Problem Statement

### Business Context

The metallurgical plant processes ~100-ton steel batches in refractory-lined ladles before continuous casting. Electricity for graphite electrode arc heating accounts for 25-35% of secondary steelmaking costs. 
Operators must maintain steel within a narrow temperature window (typically ±5–10°C). Lower temperature causes recycles and delays. Higher temperature leads to wasted energy and faster ladle wear. Current heuristic rules ('heat until safe') result in systematic overheating and unpredictable cycle times. Thus, precise heating within set time could minimize energy waste and therefore reduce production costs while ensuring quality for continuous casting.

Example Heat Cost: 300 kWh electricity at €0.08/kWh = €24 per heat. At 300 heats/month/furnace, 10% savings = €7.2K/month = €86K/year per furnace.

### ML Objective

Predict the human real age by photo based on data of the APPA-REAL dataset.


## Project Structure

```
age_by_photo/
├── data/                                       # Data files (not included in repo)
├── results/                                    # Model outputs and results
├── main_age_prediction.ipynb                   # Main source
├── requirements.txt                            # Python dependencies
├── .gitignore
└── README.md
```


## Libraries

- **Pandas** - data manipulation and analysis
- **NumPy** - numerical computations
- **Matplotlib & Seaborn** - data visualization
- **Scikit-learn** - machine learning utilities and Ridge Regression
- **PyOD** - outlier detection (KNN)
- **Optuna** - hyperparameter optimization
- **CatBoost** - gradient boosting model


## Data

The dataset contains public data  and is used for educational purposes only.

### Download Links

The data can be downloaded from the following sources and should be placed into `./data` directory:

- [data_arc_new.csv](https://drive.google.com/file/d/1Uc2WbhW9U5-TtLr8X82QQxk36J9CMNCu/view?usp=sharing)
- [data_bulk_new.csv](https://drive.google.com/file/d/1LtAejlRIp5xm736IJEcteEixNapV95pd/view?usp=sharing)
- [data_bulk_time_new.csv](https://drive.google.com/file/d/1hYBVDx2I5WtIDlDKkkmztu2ZvqKYPB9J/view?usp=sharing)
- [data_gas_new.csv](https://drive.google.com/file/d/1GSdUxiW0iKIm9r_0crC4_HjpJEi2Id1U/view?usp=sharing)
- [data_temp_new.csv](https://drive.google.com/file/d/1DE_OKJ9NvheG5x1PXzRngy4LCenpaZM-/view?usp=sharing)
- [data_wire_new.csv](https://drive.google.com/file/d/1Muzt0mFQNLYlJPoET8mm1fz2icTyOr6i/view?usp=sharing)
- [data_wire_time_new.csv](https://drive.google.com/file/d/1ibkWLU7GmAMenZkwDJ6_z-8gMbN-2kyF/view?usp=sharing)

### Dataset Descriptions

| File | Description |
|------|-------------|
| `` |  |
| `.csv` |  |


## Quick start

1. Clone the repository
2. Prepare virtual environment
    Create virtual environment
    ```bash 
    python3 -m venv venv_age
    ```
    Activate virtual environment
    ```bash
    source venv_age/bin/activate
    ```  # Linux/Mac

    or

    ```bash
    .\venv_age\Scripts\activate
    ```   # Windows
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   Go to main_age_prediction.ipynb, 
   choose .venv_age as kernel,
   run jupiter notebook. 


## Approach

1. **Exploratory Data Analysis (EDA)**: 
   - Load APPA-REAL metadata (file_name, real_age) and visualize age distribution 
   - Display sample face images and check for data quality issues like image loading errors
2. **Data Preprocessing**:
   - Read face-cropped images, resize to fixed input size
   - Normalize pixel values to or standardize using ImageNet stats
​   - Split into train/val/test sets stratified by age to preserve distribution
3. **Feature Engineering**:
   - Data augmentation pipeline (horizontal flips, rotations, shifts, zooms) applied on-the-fly via generators
4. **Model Training and Selection**:
   - Transfer learning with pre-trained ImageNet CNN backbone (ResNet/Xception/MobileNet)
   - Add global average pooling + dense layers + single regression output; freeze base layers initially, then fine-tune
   - Train with MAE loss, Adam optimizer, and callbacks (EarlyStopping, ModelCheckpoint)
6. **Model Evaluation**: 
   - Compute MAE on hold-out test set; plot predictions vs true ages and error distributions
   - Visualize sample predictions overlaid on images to inspect failure cases across age ranges


## Results

Model's test MAE is 5.9: predictions are off by 6.0°C on average.




## Futher improvements
The project is done for educational purposes only. For futher development these steps may be considered:
- Incorporate paper's insight of 'Apparent and real age estimation in still images with
deep residual regressors on APPA-REAL database' by Eirikur Agustsson, Radu Timofte, Sergio Escalera, Xavier Baró, Isabelle Guyon, Rasmus Rothe
Direct real-age regression with the residual DEX approach—train first on apparent age (using the 260k crowdsourced votes), then learn the apparent→real residual. This exploits human perception bias and typically beats direct regression.

- Address age distribution bias;
The APPA-REAL dataset is heavily skewed; implement class weighting, focal loss, or ordinal regression to boost performance on tails (children/elderly) where simple MAE underperforms.

Advanced fine-tuning: Add learning rate scheduling (cosine annealing), gradient clipping, and mixup/cutmix augmentation. Consider progressive unfreezing with differential learning rates across backbone layers.

Leverage full dataset: Use the complete apparent age distribution (not just real_age) for pre-training, then residual correction. Ensemble multiple backbones (ResNet + EfficientNet + Vision Transformer) with stacking for leaderboard-level MAE.



## Author

**Iuliia Kuznetsova**  
January 2024

