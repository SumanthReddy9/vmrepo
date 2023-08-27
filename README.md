# Pet Adoption Model

This repository contains code for training, testing, and evaluating a Pet Adoption prediction model using XGBoost.

## Setup and Usage

1. **Clone the Repository:**

    ```sh
    git clone https://github.com/SumanthReddy9/vmrepo.git
    cd vmrepo
    ```

2. **Run the Script:**

    You can run the script using the shell script with arguments. Below are the available arguments:

    - `--target_column`: Name of the target column 
    - `--early_stopping_rounds`: Early stopping rounds for training 
    - `--num_boost_round`: Maximum number of boosting rounds 

    Example usage(Ubuntu/Ubuntu Subsystem):

    ```
    chmod 777 run.sh 
    ./run.sh Adopted 10 1000
    ```

3. **View Results:**

    - After running the script, the test set performance metrics (F1 Score, Accuracy, Recall) will be displayed in the terminal.
    - The artifacts are stored 'artifacts/model/' folder
    - The output predictions of all rows are stored in 'output/' folder
     

