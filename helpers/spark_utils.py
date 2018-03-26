from pyspark.sql import SparkSession
import os
import shutil


def spark_add_py_module(spark, module, tmp_dir='/tmp'):
    """
    Recursivly add py files to Spark given a base module
    """
    root_dir = os.path.dirname(os.path.dirname(module.__file__))
    base_dir = module.__name__
    zip_file = shutil.make_archive(os.path.join(tmp_dir, base_dir), 'zip',
                                   base_dir=base_dir, root_dir=root_dir)
    spark.sparkContext.addPyFile(zip_file)
