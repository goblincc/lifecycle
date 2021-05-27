package model

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.rdd.RDD

object Doc2vec {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .config("spark.hadoop.validateOutputSpecs", value = false)
      .enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("warn")
    val sqltxt =
      s"""
         |select doc from (
         |select hdid, collect_list(doc_id) as doc from (
         |select hdid, doc_id, dt from persona.yylive_lunxun_doc_click_day group by hdid, doc_id, dt) as a group by hdid) as a where size(doc) >=2
       """.stripMargin
    val docDF = spark.sql(sqltxt)
    docDF.show(10, false)

    val word2Vec = new Word2Vec()
      .setInputCol("doc")
      .setOutputCol("result")
      .setVectorSize(5)
      .setMinCount(0)

    val model = word2Vec.fit(docDF)

    val output = "hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang"
    model.save(output + "/model")

    model.getVectors.rdd.map(p => {
      p.getString(0) + "/t" + p.getAs[DenseVector](1).values.mkString(",")
    }).repartition(1).saveAsTextFile(output + "/vec")


    spark.close()
  }
}
