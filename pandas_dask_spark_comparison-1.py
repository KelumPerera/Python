# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:54:31 2023

@author: Kelum.Perera
"""


#!pip install pyspark

# Read files using pandas ( by defaulf use only one process or thread to process the data)

# Load the GL data
%%time
path = r"C:\Users\Kelum.Perera\Downloads\data-master\nyse_all\nyse_data"
#file_identifier = "*.csv"   # If your files are text files(Values separated by Tab) use "*.txt" or (Values separated by pip (|) use ".pip"
merged_pandas_data = pd.DataFrame()
record_count = 0
header_name = ["stock_id","trans_date","open_price","low_price","high_price","close_price","volume"]
for f in glob.glob(path + "/*" ):
    print(f)
    df = pd.read_csv(f, names = header_name ,delimiter=',') #,header=None,parse_dates=True
    merged_pandas_data = merged_pandas_data.append(df,ignore_index=True)
    record_count +=  df.shape[0]

merged_pandas_data.info()
merged_pandas_data.head()
    

# Read files using Dask

merged_dask_data = dd.read_csv(r"C:\Users\Kelum.Perera\Downloads\data-master\nyse_all\nyse_data\*", names = header_name,dtype={ 'stock_id':'str','trans_date':'int64','open_price':np.float64,'low_price':np.float64,'high_price':np.float64,"close_price":np.float64,'volume':'int64'}) #, blocksize = 

merged_dask_data.shape[0].compute()
merged_dask_data.head()

merged_dask_data.npartitions
#merged_dask_data.repartition(2)



# Read files using Spark

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DecimalType, StringType, DateType, TimestampType

spark = SparkSession.builder.appName('Read CSV File into DataFrame').getOrCreate()

my_schema = StructType([
    StructField("stock_id", StringType(), True),
    StructField("trans_date", IntegerType(), True),
    StructField("open_price", DecimalType(), True),
    StructField("low_price", DecimalType(), True),
    StructField("high_price", DecimalType(), True),
    StructField("close_price", DecimalType(), True),
    StructField("volume", IntegerType(), True),
    ])

my_schema = '''stock_id STRING,trans_date STRING,open_price FLOAT,low_price FLOAT,high_price FLOAT,close_price FLOAT,volume BIGINT'''


merged_spark_data = spark.read.csv('C:/Users/Kelum desktop PC/Downloads/New_Test/*', schema = my_schema) # ,inferSchema=True, sep=',', header=False,

merged_spark_data.count()

df = merged_spark_data.toPandas()
df.head()



!pip install git+https://github.com/dsdanielpark/Bard-API.git