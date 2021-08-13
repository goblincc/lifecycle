package analysis

import model.DocCTRPredict.{modelPath, saveData2hive}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DataTypes, DoubleType, StringType, StructField}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import utils.TimeUtils

object gbt_risk_predict {

  val modelPath = "hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang/risk"

  def main(args: Array[String]): Unit = {
    val dts = args(0)
    val dt = TimeUtils.changFormat(dts)
    val spark = SparkSession.builder()
      .config("spark.hadoop.validateOutputSpecs", value = false)
      .enableHiveSupport()
      .getOrCreate()
    spark.sparkContext.setLogLevel("error")

    val sqlTxt =
      s"""
         |SELECT *,
         |    (cnt_30 + cnt_60 + cnt_90) as cnt_90_all,
         |    (day_30 + day_60 + day_90) as day_90_all,
         |    (sid_30 + sid_60 + sid_90) as sid_90_all,
         |    (sid_all_30 + sid_all_60 + sid_all_90) as sid_all_90_dr,
         |    (gift_cnt_30 + gift_cnt_60 + gift_cnt_90) as gift_cnt_90_all,
         |    (sum_30 + sum_60 + sum_90) as sum_90_all,
         |    (alldt_30 + alldt_60 + alldt_90) as alldt_90_all,
         |    (amount_30 + amount_60 + amount_90) as amount_90_all
         |FROM persona.yylive_uid_feature_info
         |WHERE dt='${dt}'
       """.stripMargin

    val data = spark.sql(sqlTxt)
//    val model_dt = TimeUtils.addDate(dts, -2)
//    println("model_dt:" + model_dt)
    val model = PipelineModel.read.load( modelPath + "/piperisk_20210803")
    val predict = model.transform(data)
    saveData2hive(spark, dt, predict)
  }

  def saveData2hive(spark:SparkSession, dt: String, dataFrame: DataFrame): Unit ={
    val structFields = Array(
      StructField("uid",StringType,true),
      StructField("prediction",DoubleType,true),
      StructField("probability",DoubleType,true)
    )
    val structType = DataTypes.createStructType(structFields)
    val row: RDD[Row] = dataFrame.select("uid", "prediction", "probability").rdd.map(p => {
      val uid = p.getString(0)
      val prediction = p.getDouble(1)
      val probability = p.getAs[DenseVector](2)(1)
      Row(uid, prediction, probability)
    })

    spark.createDataFrame(row,structType).createOrReplaceTempView("tb_save")

    spark.sql(
      s"""
         |insert overwrite table persona.yylive_uid_feature_predict partition(dt='${dt}')
         |	select * from tb_save
       """.stripMargin)
  }

}
