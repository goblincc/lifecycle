package graphx

import org.apache.spark.sql.SparkSession

object demo {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .config("spark.hadoop.validateOutputSpecs", value = false)
      .master("local[2]")
      .enableHiveSupport()
      .getOrCreate()
    val sc = spark.sparkContext
    val rdd1 = sc.makeRDD(Array(("1","Spark"),("2","Hadoop"),("3","Scala"),("4","Java")),2)
    val rdd3 = sc.makeRDD(Array(("Spark","1"),("Hadoop","2"),("Scala","3"),("Java","4")),2)
    val rdd2 = sc.makeRDD(Array(("1","30K"),("2","15K"),("3","25K"),("5","10K")),2)

    rdd1.join(rdd2).collect.foreach(println)
    rdd1.join(rdd3).collect.foreach(println)
  }
}
