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
    val data_dt = TimeUtils.addDate(dts, -1)
    val data_dt2 = TimeUtils.addDate2(dts, -1)
    val spark = SparkSession.builder()
      .config("spark.hadoop.validateOutputSpecs", value = false)
      .enableHiveSupport()
      .getOrCreate()
    spark.sparkContext.setLogLevel("error")

    val sqlTxt =
      s"""
         |select a.*,
         |	  cast(is_nick_modify as string) as is_nick_modifys,
         |    datediff('${data_dt}',regtime) AS reg_day,
         |    max_IP_cnt as max_ip_cnts,
         |    avg_IP_cnt as avg_ip_cnts,
         |    stdev_IP_cnt as stdev_ip_cnts,
         |    nvl(chid_90/alldt_90,0) AS avg_chid_times_90,
         |    nvl(paymethod_90/alldt_90,0) AS avg_method_times_90,
         |    nvl(buyerid_cnt/alldt_90,0) AS buyer_pay_ratio
         |from (
         |select * from persona.yylive_uid_feature_info WHERE dt='${data_dt}') as a
         |inner join (
         |select userid  FROM persona.ybpay_paygate_order_d WHERE dt = '${data_dt2}' group by userid) as b
         |on a.uid = b.userid
         |where cast(a.uid as bigint) > 0
       """.stripMargin
    println("sqlTxt:", sqlTxt)
    val data = spark.sql(sqlTxt)
//    val model_dt = TimeUtils.addDate(dts, -2)
//    println("model_dt:" + model_dt)
//    piperisk_20210814
    val model = PipelineModel.read.load( modelPath + "/piperisk_20210827")
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
