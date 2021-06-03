package model

import model.DocVecCTR.registUDF
import model.GbtPredict.{modelPath, saveData2hive}
import org.apache.spark.ml.{PipelineModel, linalg}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{DataTypes, DoubleType, StringType, StructField}
import utils.TimeUtils

import scala.collection.mutable

object DocPredict {
  def main(args: Array[String]): Unit = {
    val dt = TimeUtils.changFormat(args(0))
    val spark = SparkSession.builder().enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("warn")

    val doc2Int: collection.Map[String, Int] = spark.sql(s"SELECT * FROM persona.yylive_dws_doc_index  WHERE dt='${dt}'")
      .rdd.map(p => {
      (p.getAs[String](0), p.getAs[Int](1))
    }).collectAsMap()

    val alsModel: ALSModel = ALSModel.read.load("hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang/ALS/AlsModel_"+ dt)

    val intToVector: collection.Map[Int, linalg.Vector] = alsModel.itemFactors.rdd.map(p => {
      val idx = p.getAs[Int](0)
      val arr = p.getAs[mutable.WrappedArray[Float]](1)
      val vector: linalg.Vector = Vectors.dense(
        arr.map(_.toDouble).toArray
      )
      (idx, vector)
    }).collectAsMap()

    registUDF(spark,doc2Int,intToVector)

    val sqlTxt =
      s"""
         |select *, docVec2(doc_id) as docVec from persona.yylive_dws_user_docid_remain_predict  WHERE dt='${dt}'
       """.stripMargin
    print("sqlTxt:", sqlTxt)

    val data = spark.sql(sqlTxt)
    val model = PipelineModel.read.load( modelPath + "/pipelineCTR_" + dt )
    val predict = model.transform(data)
    predict.show(5, false)
    saveData2hive(spark, dt, predict)

  }

  def saveData2hive(spark:SparkSession, dt: String, dataFrame: DataFrame): Unit ={
    val structFields = Array(
      StructField("hdid",StringType,true),
      StructField("docid",StringType,true),
      StructField("prediction",DoubleType,true),
      StructField("probability",DoubleType,true)
    )
    val structType = DataTypes.createStructType(structFields)
    val row: RDD[Row] = dataFrame.select("hdid", "doc_id","prediction", "probability").rdd.map(p => {
      val hdid = p.getString(0)
      val docid = p.getString(1)
      val prediction = p.getDouble(2)
      val probability = p.getAs[DenseVector](3)(1)
      Row(hdid, docid, prediction, probability)
    })

    spark.createDataFrame(row,structType).createOrReplaceTempView("tb_save")

    spark.sql(
      s"""
         |insert overwrite table persona.yylive_dws_user_doc_predict_info partition(dt='${dt}')
         |	select * from tb_save
       """.stripMargin)

  }

  def registUDF(spark: SparkSession, doc2Int: collection.Map[String, Int], intToVector: collection.Map[Int, linalg.Vector]): Unit = {
    spark.udf.register("docVec2", (s: String) => {
      val idx = doc2Int.getOrElse(s, -1)
      intToVector.getOrElse(idx, Vectors.dense(Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    })

  }

}
