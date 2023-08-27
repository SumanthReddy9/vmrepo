import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, recall_score
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import argparse


# sudo pip3 install gcsfs


class PetAdoptionModel:
    def __init__(self, model_dir='artifacts/model'):
        """
        Initialize the PetAdoptionModel class.

        Args:
            model_dir (str): Directory to save the trained model.
        """
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None

    def load_dataset(self, url):
        """
        Load the dataset from a given URL into a Pandas DataFrame.

        Args:
            url (str): URL of the dataset.
        """
        self.df = pd.read_csv(url)

    def feature_engineering(self, target_column='Adopted'):
        """
            Feature engineering function.

            Args:
                self: The object instance.
                target_col (str): The name of the target column.

            Returns:
                None.

            """

        missing_data = self.df.isnull().sum() / len(self.df)
        missing_data = missing_data[missing_data > 0]
        missing_data_df = missing_data.to_frame()
        missing_data_df.columns = ['Count']
        missing_data_df.index.names = ['Names']
        missing_data_df["Names"] = missing_data_df.index
        drop_cols = missing_data_df[missing_data_df["Count"] > 0.5]["Names"].tolist()
        if target_column in drop_cols:
            drop_cols.remove(target_column)
        self.df = self.df.drop(drop_cols, axis=1)
        miss_dict = dict(missing_data)

        for x in list(miss_dict.keys()):
            if x not in drop_cols:
                if self.df[x].dtype == 'object':
                    self.df[x] = self.df[x].fillna(
                        list(dict(self.df[x].value_counts()).keys())[0])
                else:
                    self.df[x] = self.df[x].fillna(self.df[x].mean())

    def label_encoder(self, data = None):
        if data is None:
            for x in self.df.columns:
                if self.df[x].dtype == 'object':
                    lbl = preprocessing.LabelEncoder()
                    lbl.fit(self.df[x].values)
                    self.df[x] = lbl.transform(self.df[x].values)
        else:
            for x in data.columns:
                if data[x].dtype == 'object':
                    lbl = preprocessing.LabelEncoder()
                    lbl.fit(data[x].values)
                    data[x] = lbl.transform(data[x].values)
            return data

    def split_dataset(self, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2, target_column = 'Adopted'):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            train_ratio (float): Ratio of the dataset to use for training.
            validation_ratio (float): Ratio of the dataset to use for validation.
            test_ratio (float): Ratio of the dataset to use for testing.
            target_column (str): Target Column of which we are going to predict
        """
        train_df, temp_df = train_test_split(self.df, test_size=1 - train_ratio, random_state=42)
        validation_df, test_df = train_test_split(temp_df, test_size=test_ratio / (test_ratio + validation_ratio),
                                                  random_state=42)

        self.X_train = train_df.drop(columns=[target_column])
        self.y_train = train_df[target_column]
        self.X_val = validation_df.drop(columns=[target_column])
        self.y_val = validation_df[target_column]
        self.X_test = test_df.drop(columns=[target_column])
        self.y_test = test_df[target_column]

    def train_model(self, early_stopping_rounds=50, num_boost_round=10000):
        """
        Train an XGBoost model with early stopping.

        Args:
            early_stopping_rounds (int): Number of rounds to perform early stopping.
            num_boost_round (int): Maximum number of boosting rounds.

        Returns:
            None
        """
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist'
        }

        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)

        eval_results = {}
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            evals=[(dval, 'validation')],
            evals_result=eval_results,
            verbose_eval=True
        )

    def test_model(self):
        """
        Test the trained model on the test set.

        Returns:
            tuple: F1 score, accuracy, and recall.
        """
        dtest = xgb.DMatrix(self.X_test)
        y_pred = self.model.predict(dtest)
        y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]

        self.f1 = f1_score(self.y_test, y_pred_binary)
        self.accuracy = accuracy_score(self.y_test, y_pred_binary)
        self.recall = recall_score(self.y_test, y_pred_binary)

        return self.f1, self.accuracy, self.recall

    def save_model(self, model_name='xgb_model.model'):
        """
        Save the trained model to a specified file.

        Args:
            model_name (str): Name of the model file.

        Returns:
            str: Path to the saved model.
        """
        model_path = os.path.join(self.model_dir, model_name)
        self.model.save_model(model_path)
        return model_path

    def load_model(self, model_path):
        """
        Load a trained model from a specified file.

        Args:
            model_path (str): Path to the model file.

        Returns:
            None
        """
        self.model = xgb.Booster(model_file=model_path)

    def predict(self, input_data):
        """
        Perform inference using the trained model on new data.

        Args:
            input_data (array-like or DataFrame): Input data for inference.

        Returns:
            array: Predicted probabilities.
        """
        dinput = xgb.DMatrix(input_data)
        predictions = self.model.predict(dinput)
        predicted_classes = [1 if p >= 0.5 else 0 for p in predictions]
        return predicted_classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_column', type = str, required=False, default='Adopted')
    parser.add_argument('--early_stopping_rounds', type=int, required=False, default=50)
    parser.add_argument('--num_boost_round', type=int, required=False, default=10000)
    args = parser.parse_args()
    target_column = args.target_column
    early_stopping_rounds = args.early_stopping_rounds
    num_boost_round = args.num_boost_round

    # Task1
    model = PetAdoptionModel()
    dataset_url = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
    model.load_dataset(dataset_url)
    model.feature_engineering(target_column=target_column)
    model.label_encoder()
    model.split_dataset()
    model.train_model(early_stopping_rounds, num_boost_round)
    f1, accuracy, recall = model.test_model()

    print("Test Set Performance:")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")

    saved_model_path = model.save_model()
    print(f"Model saved at: {saved_model_path}")

    # Task2
    model.load_model(saved_model_path)
    data = pd.read_csv(dataset_url)
    data_processed = data.copy()
    data_processed = model.label_encoder(data_processed)
    all_predictions = model.predict(data_processed.drop(columns=[target_column]))
    all_predictions = ['Yes' if x == 1 else 'No' for x in all_predictions]
    data[target_column + '_prediction'] = all_predictions
    os.makedirs('output', exist_ok=True)
    data.to_csv('output/results.csv', index=False)
