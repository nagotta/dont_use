import numpy as np
import pandas as pd
list1=[[1,2,3], [21,22,23], [31,32,33]]
index1 = ["Row1", "Row2", "Row3"]
columns1 =["Col1", "Col2", "Col3"]
pd.DataFrame(data=list1, index=index1, columns=columns1)