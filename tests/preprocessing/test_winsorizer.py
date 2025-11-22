import numpy as np
import pytest
import pandas as pd

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer

@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

#ARRANGE

    #generate example data 
    rng= np.random.RandomState(42)
    X = pd.DataFrame({"x": rng.normal(0, 1, 1000)})

    #define the transformer 
    transformer = Winsorizer(columns=["x"], lower_quantile=lower_quantile, upper_quantile=upper_quantile)

#ACT
    transformer.fit(X)
    X_transform = transformer.transform(X) #here we have avoided modifying the original data 

#ASSERT
    #caclulate what you would expect to find from the transformer 
    lower_value = transformer.lower_quantile_["x"]
    upper_value = transformer.upper_quantile_["x"]

    #check that all transformed values are within the quantile bounds
    assert X_transform["x"].min() >=lower_value
    assert X_transform["x"].max() <=upper_value

    #check that transformation did not change shape or introduce missing values (NaN)
    #for this check, you would have needed to define the transformed and original data as seperate
