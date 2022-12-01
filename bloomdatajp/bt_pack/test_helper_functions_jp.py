import random
import numpy as np
import pandas as pd
from bloomdatajp.helper_functions_jp import null_count, train_test_split
from bloomdatajp.helper_functions_jp import randomize, list_2_series
# Test cases
df = pd.DataFrame(np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]]))
null_df_1 = pd.DataFrame(
    np.array([[10, np.nan, 12], [13, np.nan, 15], [16, 17, 18]])
)
null_df_2 = pd.DataFrame(
    np.array([[np.nan, np.nan, np.nan], [13, 14, 15], [16, 17, 18]])
)


def test_null_count():
    """
    Testing null_count function.
    """

    assert null_count(df) == 0
    assert null_count(null_df_1) == 2
    assert null_count(null_df_2) == 3


def test_train_test_split():
    """
    Testing train_test_split function.
    """

    train_1, test_1 = train_test_split(df, frac=0.8)
    train_2, test_2 = train_test_split(df, frac=0.2)
    train_3, test_3 = train_test_split(null_df_1, frac=0.8)

    assert len(train_1) == 2
    assert len(test_1) == 1

    assert len(train_2) == 1
    assert len(test_2) == 2

    assert len(train_3) == 2
    assert len(test_3) == 1


def test_randomize():
    """
    Testing randomize function.
    """

    randomized_df = randomize(df, seed=10)
    assert isinstance(randomized_df, pd.DataFrame)
    assert len(randomized_df) == len(df)
    assert not randomized_df.equals(df)


def test_list_2_series():
    """
    Testing list_2_series function
    """

    random_list = random.sample(range(0, 100), 100)
    df = list_2_series(random_list, pd.DataFrame())

    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 1

    # value_tester = random.choice(random_list)
    # assert df['list'].isin([value_tester]).any()

    for value in random_list:
        assert df['list'].isin([value]).any()
