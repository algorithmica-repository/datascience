import pandas as pd
import numpy as np

ibm_stock = [10,12,15,15,10,13]
ms_stock = [20,25,30,20,15,22]

stocks = pd.DataFrame({'ibm_stock':ibm_stock, 'ms_stock':ms_stock})
stocks['ibm_stock'].std()
stocks['ms_stock'].std()

stocks['ibm_stock'].mad()
stocks['ms_stock'].mad()

np.cov(ibm_stock, ms_stock)
np.corrcoef(ibm_stock, ms_stock)

oracle_stock = [100,110,120,130,140,150]
np.cov(ibm_stock, oracle_stock)
np.corrcoef(ibm_stock, oracle_stock)

google_stock = [200,210,220,230,240,250]
np.cov(oracle_stock, google_stock)
np.corrcoef(oracle_stock, google_stock)

amazon_stock = [250,240,230,220,210,200]
np.cov(google_stock, amazon_stock)
np.corrcoef(google_stock, amazon_stock)

np.corrcoef(google_stock, amazon_stock)
