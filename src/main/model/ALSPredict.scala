package model

import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.sql.SparkSession

object ALSPredict {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("warn")

    val doc2Int: collection.Map[String, Int] = spark.sql("SELECT * FROM persona.yylive_dws_doc_index  WHERE dt='2021-05-28'")
      .rdd.map(p => {
      (p.getAs[String](0), p.getAs[Int](1))
    }).collectAsMap()

    val userToInt: collection.Map[String, Int] = spark.sql("SELECT * FROM persona.yylive_dws_user_index  WHERE dt='2021-05-28'")
      .rdd.map(p => {
      (p.getAs[String](0), p.getAs[Int](1))
    }).collectAsMap()

    registUDF(spark, doc2Int, userToInt)
    val alsModel: ALSModel = ALSModel.read.load("hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang/ALS/AlsModel")

    val sqltxt =
      s"""
         |select userUDF(hdid) as userId, docUDF(docid) as itemId from (
         |SELECT hdid FROM persona.yylive_lunxun_doc_hdid_score  WHERE dt="2021-05-28") as a,
         |(select docid from persona.yylive_lunxun_doc_hdid_score  WHERE dt="2021-05-28" group by docid) as b
       """.stripMargin

    val data = spark.sql(sqltxt)

    val dataFrame = alsModel.transform(data)

    dataFrame.show(5, false)
  }

  def registUDF(spark: SparkSession, userToInt: collection.Map[String, Int],doc2Int: collection.Map[String, Int]): Unit ={
    spark.udf.register("userUDF", (s: String) => {
      userToInt.getOrElse(s, -1)
    })

    spark.udf.register("docUDF", (s: String) => {
      doc2Int.getOrElse(s, -1)
    })

  }


}
