import pandas as pd
import model
import unittest.mock as mock

def test_test():
    assert True


@mock.patch("model.get_cols_to_dummy")
@mock.patch("model.get_cols_to_drop")
def test_feature_preprocessing(mock_cols_to_drop,
                               mock_cols_to_dummy):
    mock_cols_to_dummy.return_value = []
    mock_cols_to_drop.return_value = ['strawberry']
    df_dict = {'chocolate':[1, 2],
               'vanilla':[3,4],
               'strawberry':[5,6]}
    fake_df = pd.DataFrame(df_dict)
    fake_df, transformers = model.feature_preprocessing(fake_df)
    assert "strawberry" not in fake_df.columns.values
