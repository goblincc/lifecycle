package model

import org.apache.spark.ml.{PipelineModel, linalg}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{DataTypes, DoubleType, StringType, StructField}
import utils.TimeUtils

import scala.collection.mutable

object DocCTRPredict {

  val modelPath = "hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang/pipe"

  def main(args: Array[String]): Unit = {
    val dt = TimeUtils.changFormat(args(0))
    val spark = SparkSession.builder().enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("warn")

    val sqlTxt =
      s"""
         |select * from persona.yylive_dws_user_docid_remain_predict  WHERE dt='${dt}'
       """.stripMargin
    print("sqlTxt:", sqlTxt)

    val data = spark.sql(sqlTxt)
    val model = PipelineModel.read.load( modelPath + "/pipeDocCTR_" + dt )
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


}
