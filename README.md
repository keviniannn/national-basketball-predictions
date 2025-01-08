# NBA Cup Prediction Model

This project uses a machine learning model to predict the outcome of an NBA Cup bracket, simulating the entire tournament round by round and determining the ultimate champion.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Example Output](#example-output)
8. [Future Improvements](#future-improvements)
9. [License](#license)

---

## Overview

The NBA Cup Prediction Model simulates an in-season tournament bracket for the NBA using historical and current team statistics. It predicts the outcome of each match using a machine learning model trained on previous game data, providing results for every round until the final champion is determined.

---

## Features

- Predict outcomes of individual NBA matches based on team statistics.
- Simulate entire tournament brackets recursively, filling out each round.
- Provide round-by-round outputs for detailed analysis.
- Easy-to-use interface with customizable input.

---

## Technologies Used

- **Python**: Programming language used for development.
- **scikit-learn**: Machine learning library for training and evaluating the model.
- **pandas**: Data manipulation library for handling team and game data.
- **Jupyter Notebook**: For data exploration and model development.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/nba-cup-prediction.git
   cd nba-cup-prediction
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the necessary datasets:
   - **Team stats dataset**: `team_stats.csv`
   - **Game data dataset**: `game_data.csv`

4. Place these files in the project directory.

---

## Usage

1. **Train the Model**:
   Run the training script to prepare the machine learning model:
   ```bash
   python train_model.py
   ```

2. **Simulate the Bracket**:
   Use the pre-trained model to simulate an NBA Cup bracket:
   ```bash
   python simulate_bracket.py
   ```

3. **Customize Input**:
   Update the `first_round_matches` list in `simulate_bracket.py` with the starting teams for your tournament.

4. **View Results**:
   The script will print round-by-round predictions and the ultimate champion.

---

## Project Structure

```plaintext
nba-cup-prediction/
│
├── data/
│   ├── team_stats.csv          # Team statistics
│   ├── game_data.csv           # Historical game data
│
├── model/
│   ├── nba_prediction_model.pkl  # Trained machine learning model
│
├── scripts/
│   ├── train_model.py          # Script to train the machine learning model
│   ├── simulate_bracket.py     # Script to simulate the tournament bracket
│
├── notebooks/
│   ├── data_analysis.ipynb     # Jupyter notebook for data exploration
│   ├── model_development.ipynb # Notebook for developing the model
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Example Output

```plaintext
--- Round 1 ---
Oklahoma City Thunder vs. Dallas Mavericks -> Oklahoma City Thunder wins
Houston Rockets vs. Golden State Warriors -> Houston Rockets wins
Milwaukee Bucks vs. Orlando Magic -> Milwaukee Bucks wins
New York Knicks vs. Atlanta Hawks -> Atlanta Hawks wins

--- Semifinals ---
Oklahoma City Thunder vs. Houston Rockets -> Oklahoma City Thunder wins
Milwaukee Bucks vs. Atlanta Hawks -> Milwaukee Bucks wins

--- Final ---
Oklahoma City Thunder vs. Milwaukee Bucks -> Milwaukee Bucks wins

The predicted NBA Cup Champion is: Milwaukee Bucks
```

---

## Future Improvements

- Integrate live game data using the NBA API for real-time predictions.
- Visualize the bracket progression using libraries like `matplotlib` or `plotly`.
- Allow web-based interaction using Flask or FastAPI for ease of use.

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.
