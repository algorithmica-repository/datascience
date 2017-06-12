import numpy as np
import pandas as pd

coin_tosses = np.random.randint(0,2,1000)
pd.crosstab(index=coin_tosses, columns="count")

dice_throws = np.random.randint(1,7,100000)
pd.crosstab(index=dice_throws, columns="count")
