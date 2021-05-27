package model

import model.DocCTR.doc2vecPath
import model.GbdtTrain.saveData2hive
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types._
import utils.TimeUtils

class GbtPredict {

  val modelPath = "hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang/pipe"

  def main(args: Array[String]): Unit = {
    val dt = args(0)
    val sparkSession = SparkSession.builder().enableHiveSupport().getOrCreate()
    sparkSession.sparkContext.setLogLevel("warn")

    val sqlTxt =
      s"""
         |select * from persona.yylive_dws_user_lifecycle_predict_feature  WHERE dt = '${dt}'
       """.stripMargin
    val data = sparkSession.sql(sqlTxt)
    val model_dt = TimeUtils.addDate(dt, -10)
    val model = PipelineModel.read.load( modelPath + "/pipeline_" + model_dt)

    val predict = model.transform(data)

    saveData2hive(sparkSession, dt, predict)

  }

  def saveData2hive(spark:SparkSession, dt: String, dataFrame: DataFrame): Unit ={
    val structFields = Array(
      StructField("hdid",StringType,true),
      StructField("prediction",DoubleType,true),
      StructField("probability",DoubleType,true)
    )
    val structType = DataTypes.createStructType(structFields)
    val row: RDD[Row] = dataFrame.select("hdid", "prediction", "label", "probability").rdd.map(p => {
      val hdid = p.getString(0)
      val prediction = p.getDouble(1)
      val probability = p.getAs[DenseVector](3)(1)
      Row(hdid, prediction, probability)
    })

    spark.createDataFrame(row,structType).createOrReplaceTempView("tb_save")

    spark.sql(
      s"""
         |insert overwrite table persona.yylive_dws_user_life_predict_info partition(dt='${dt}')
         |	select * from tb_save
       """.stripMargin)

  }
}
