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
         |    cast(is_nick_modify as string) as is_nick_modifys,
         |    nvl(alldt_30/dtcnt_30,0) AS avg_pay_times_30,
         |    nvl(alldt_60/dtcnt_60,0) AS avg_pay_times_60,
         |    nvl(alldt_90/dtcnt_90,0) AS avg_pay_times_90,
         |    nvl(chid_30/alldt_30,0) AS avg_chid_times_30,
         |    nvl(chid_60/alldt_60,0) AS avg_chid_times_60,
         |    nvl(chid_90/alldt_90,0) AS avg_chid_times_90,
         |    nvl(paymethod_30/alldt_30,0) AS avg_method_times_30,
         |    nvl(paymethod_60/alldt_60,0) AS avg_method_times_60,
         |    nvl(paymethod_90/alldt_90,0) AS avg_method_times_90,
         |    nvl(buyerid_cnt/alldt_90,0) AS buyer_pay_ratio
         |FROM persona.yylive_uid_feature_info
         |WHERE dt='${dt}'
       """.stripMargin

    val data = spark.sql(sqlTxt)
//    val model_dt = TimeUtils.addDate(dts, -2)
//    println("model_dt:" + model_dt)
    val model = PipelineModel.read.load( modelPath + "/piperisk_20210814")
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
