import pandas as pd

ibm_stock = [10,12,15,15,10,13]
ms_stock = [20,25,30,20,15,22]

stocks = pd.DataFrame({'ibm_stock':ibm_stock, 'ms_stock':ms_stock})
stocks['ibm_stock'].std()
stocks['ms_stock'].std()

stocks['ibm_stock'].mad()
stocks['ms_stock'].mad()



