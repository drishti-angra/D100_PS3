import numpy as np
import pandas as pd


def load_transform():
    """Load and transform data from OpenML.

    Source: https://glum.readthedocs.io/en/latest/tutorials/glm_french_motor_tutorial/glm_french_motor.html#

    Summary of transformations:

    1. We cut the number of claims to a maximum of 4, as is done in the case study paper
       (Case-study authors suspect a data error. See section 1 of their paper for
       details).
    2. We cut the exposure to a maximum of 1, as is done in the case study paper
       (Case-study authors suspect a data error. See section 1 of their paper for
       details).
    3. We define ``'ClaimAmountCut'`` as the the claim amount cut at 100'000 per
       single claim (before aggregation per policy). Reason: For large claims,
       extreme value theory might apply. 100'000 is the 0.9984 quantile, any claims
       larger account for 25% of the overall claim amount. This is a well known
       phenomenon for third-party liability.
    4. We aggregate the total claim amounts per policy ID and join them to
       ``freMTPL2freq``.
    5. We fix ``'ClaimNb'`` as the claim number with claim amount greater zero.
    6. ``'VehPower'``, ``'VehAge'``, and ``'DrivAge'`` are clipped and/or digitized
       into bins so they can be used as categoricals later on.
    """

    # categorical variable means that it is treated as a discrete category or group instead of as a continuous numeric value 
         #hence, instead of ages as a variable, use the variable Young, Middle Aged, Old 



    # load the datasets
    # first row (=column names) uses "", all other rows use ''
    # use '' as quotechar as it is easier to change column names
    df = pd.read_csv(
        "https://www.openml.org/data/get_csv/20649148/freMTPL2freq.arff", quotechar="'"
    )

    # rename column names '"name"' => 'name'
    df = df.rename(lambda x: x.replace('"', ""), axis="columns")

    #make sure all values in the IDpol column is an integer 
    df["IDpol"] = df["IDpol"].astype(np.int64)

    # moves IDpol column from being a regular column to being a DataFrame's index. 
    # Inplade = True modifies the original DataFrame rather than returning a new one 
    df.set_index("IDpol", inplace=True)

    #uploading a new dataset called df_sev. this contains data on the claim amounts per IDpol.
    df_sev = pd.read_csv(
        "https://www.openml.org/data/get_csv/20649149/freMTPL2sev.arff", index_col=0
    )

    # join ClaimAmount from df_sev to df:
    #   1. cut ClaimAmount at 100_000
    #   2. aggregate ClaimAmount per IDpol
    #   3. join by IDpol

    #this creates a new column called ClaimAmountCut where if ClaimAmount > 10000, it is replaced with 10,000
    df_sev["ClaimAmountCut"] = df_sev["ClaimAmount"].clip(upper=100_000)

    #this joins the data in df_sev with the data in df and it is combined by the index of both dataframes (the IDpol) which is level = 0 
    df = df.join(df_sev.groupby(level=0).sum(), how="left")

    #after joining, some IDPol have no ClaimAmount (will show as NaN)
    # hence, replace NaN with 0 as the ClaimAmount and the ClaimAmountCut 
    # inplace=True modifies the df dataframe directly
    df.fillna(value={"ClaimAmount": 0, "ClaimAmountCut": 0}, inplace=True)

    # Note: Zero claims must be ignored in severity models,
    # because the support is (0, inf) not [0, inf).
    #if the ClaimAmount <=0 AND the number of claims occured is >=1 (CLaimNB >=1), then we must set the no of claims occured at 0 since this is a contradiction
    df.loc[(df.ClaimAmount <= 0) & (df.ClaimNb >= 1), "ClaimNb"] = 0

    # correct for unreasonable observations (that might be data error)
    # see case study paper

    #if the PolID has > 4 claims, set it to 4 claims (correcting for outliers). Limiting ClaimNb to 4 
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)

    #limiting Exposure to 1 (exposure is the fraction of a year which the policy was active for)
    df["Exposure"] = df["Exposure"].clip(upper=1)

    # Clip and/or digitize predictors into bins

    #cap Vehicle Power to 9
    df["VehPower"] = np.minimum(df["VehPower"], 9)

    #vehicles are binned 
    #first, if the vehicle age is 10, change it to 9 (maybe a vehicle age of 10 is an error)
    #next, create 2 bins: 
         #1. bin which is <=1 
         #2. bin for 1<x<=10 
    df["VehAge"] = np.digitize(
        np.where(df["VehAge"] == 10, 9, df["VehAge"]), bins=[1, 10]
    )

    #creating bins for drivers age (into age brakets rather than age- continuous to categorical variable)
         #1. <=21
         #2. between 22-26
         #3. between 27-31
         #4. between 33-41
         #5. between 42-51
         #6. between 52-71


    df["DrivAge"] = np.digitize(df["DrivAge"], bins=[21, 26, 31, 41, 51, 71])

   #changing IDpol back to a regular column and returning to the default index 
    df = df.reset_index()

    return df
