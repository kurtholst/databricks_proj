# Databricks notebook source
# Filepath for iris dataset in DBFS
#display(dbutils.fs.ls("/databricks-datasets/Rdatasets/data-001/csv/datasets/iris.csv"))
display(dbutils.fs.ls("/databricks-datasets/Rdatasets/data-001/csv/datasets/"))


# COMMAND ----------

iris.head()