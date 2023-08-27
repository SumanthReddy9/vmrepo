import unittest
import pandas as pd
from src.TaskOne import PetAdoptionModel


class TestPetAdoptionModel(unittest.TestCase):
    def setUp(self):
        self.model = PetAdoptionModel()
        self.dataset_url = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
        self.model.load_dataset(self.dataset_url)
        self.model.feature_engineering(target_column='Adopted')
        self.model.label_encoder()
        self.model.split_dataset()
        self.model.train_model()

    def test_feature_engineering(self):
        # Modify the dataset to include missing values
        self.model.df.loc[1, 'Gender'] = None
        total_rows = len(self.model.df)
        self.model.df.loc[:1 + total_rows // 2, 'Age'] = None
        self.model.feature_engineering()

        self.assertFalse('Age' in self.model.df.columns)
        self.assertTrue('Gender' in self.model.df.columns)
        self.assertTrue('Type' in self.model.df.columns)
        self.assertTrue('Breed1' in self.model.df.columns)

        # Assert that missing values are properly filled
        self.assertFalse(self.model.df.isnull().any().any())

    def test_label_encoder_with_data(self):
        # Create a new DataFrame with categorical values
        data = pd.DataFrame({'Type': ['Dog', 'Cat', 'Dog']})

        transformed_data = self.model.label_encoder(data)

        # Check that the transformed data has the 'Type' column transformed
        self.assertTrue('Type' in transformed_data.columns)
        self.assertTrue(all(isinstance(val, int) for val in transformed_data['Type']))

    def test_train_model(self):
        self.assertIsNotNone(self.model.model)

    def test_test_model(self):
        f1, accuracy, recall = self.model.test_model()
        self.assertGreaterEqual(f1, 0)
        self.assertGreaterEqual(accuracy, 0)
        self.assertGreaterEqual(recall, 0)

    def test_save_load_model(self):
        saved_model_path = self.model.save_model()
        self.assertTrue(saved_model_path)
        loaded_model = PetAdoptionModel()
        loaded_model.load_model(saved_model_path)
        self.assertIsNotNone(loaded_model.model)

    def test_predict(self):
        input_data = pd.DataFrame(self.model.X_test.iloc[0]).T
        predictions = self.model.predict(input_data)
        self.assertTrue(all(pred in [0, 1] for pred in predictions))


if __name__ == "__main__":
    unittest.main()
