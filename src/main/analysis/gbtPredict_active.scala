package analysis

import model.DocCTRPredict.{modelPath, saveData2hive}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DataTypes, DoubleType, StringType, StructField}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import utils.TimeUtils

object gbtPredict_active {

  def main(args: Array[String]): Unit = {
    val dts = args(0)
    val dt = TimeUtils.changFormat(dts)
    val spark = SparkSession.builder().enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("warn")

    val sqlTxt =
      s"""
         |select a.*,
         |if(event_act_list_d4_d7 is null, '', event_act_list_d4_d7 ) as event_act_list_d4_d7,
         |if(event_act_list_d8_d14 is null, '', event_act_list_d8_d14) as event_act_list_d8_d14,
         |if(event_act_list_d15_d30 is null, '', event_act_list_d15_d30) as event_act_list_d15_d30
         |from (
         |select
         |hdid,
         |bd_consum,
         |bd_marriage,
         |bd_subconsum,
         |bd_age,
         |sex,
         |city_level,
         |sjp,
         |sys,
         |start_cnt_d7,
         |start_cnt_d14,
         |start_cnt_d30,
         |active_days_d7,
         |active_days_d14,
         |active_days_d30,
         |total_watch_dr_d7,
         |total_watch_dr_d14,
         |total_watch_dr_d30,
         |consume_cnt_d7,
         |consume_cnt_d14,
         |consume_cnt_d30,
         |push_click_cnt_d7,
         |push_click_cnt_d14,
         |push_click_cnt_d30,
         |if(applist is null, '', applist) as applists,
         |if(age is null, 'other', age) as yy_age,
         |1 as is_exposure_v2
         |from persona.yylive_baoxiang_feature_day_predict where dt = '2021-07-14') as a
         |left join (
         |SELECT hdid,
         |       event_act_list_d4_d7,
         |       event_act_list_d8_d14,
         |       event_act_list_d15_d30,
         |       dt
         |FROM persona.yylive_dws_web_event_act_d
         |WHERE dt = '2021-07-14') as b
         |on a.hdid = b.hdid
       """.stripMargin
    print("sqlTxt:", sqlTxt)

    val data = spark.sql(sqlTxt)


    val sqlTxt2 =
      s"""
         select a.*,
         |if(event_act_list_d4_d7 is null, '', event_act_list_d4_d7 ) as event_act_list_d4_d7,
         |if(event_act_list_d8_d14 is null, '', event_act_list_d8_d14) as event_act_list_d8_d14,
         |if(event_act_list_d15_d30 is null, '', event_act_list_d15_d30) as event_act_list_d15_d30
         |from (
         |select
         |hdid,
         |bd_consum,
         |bd_marriage,
         |bd_subconsum,
         |bd_age,
         |sex,
         |city_level,
         |sjp,
         |sys,
         |start_cnt_d7,
         |start_cnt_d14,
         |start_cnt_d30,
         |active_days_d7,
         |active_days_d14,
         |active_days_d30,
         |total_watch_dr_d7,
         |total_watch_dr_d14,
         |total_watch_dr_d30,
         |consume_cnt_d7,
         |consume_cnt_d14,
         |consume_cnt_d30,
         |push_click_cnt_d7,
         |push_click_cnt_d14,
         |push_click_cnt_d30,
         |if(applist is null, '', applist) as applists,
         |if(age is null, 'other', age) as yy_age,
         |0 as is_exposure_v2
         |from persona.yylive_baoxiang_feature_day_predict where dt = '2021-07-14') as a
         |left join (
         |SELECT hdid,
         |       event_act_list_d4_d7,
         |       event_act_list_d8_d14,
         |       event_act_list_d15_d30,
         |       dt
         |FROM persona.yylive_dws_web_event_act_d
         |WHERE dt = '2021-07-14') as b
         |on a.hdid = b.hdid
       """.stripMargin
    print("sqlTxt2:", sqlTxt2)

    val data2 = spark.sql(sqlTxt2)

    val model = PipelineModel.read.load("hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang/pipe/pipelineActive_" + dt )
    val predict = model.transform(data)
    val predict2 = model.transform(data2)
    saveData2hive(spark, dt, predict,"1")
    saveData2hive(spark, dt, predict2, "0")
  }

  def saveData2hive(spark:SparkSession, dt: String, dataFrame: DataFrame, dtype: String): Unit ={
    val structFields = Array(
      StructField("hdid",StringType,true),
      StructField("is_exposure_v2",StringType,true),
      StructField("prediction",DoubleType,true),
      StructField("probability",DoubleType,true)
    )
    val structType = DataTypes.createStructType(structFields)
    val row: RDD[Row] = dataFrame.select("hdid", "is_exposure_v2","prediction", "probability").rdd.map(p => {
      val hdid = p.getString(0)
      val docid = p.getAs[Int](1).toString
      val prediction = p.getDouble(2)
      val probability = p.getAs[DenseVector](3)(1)
      Row(hdid, docid, prediction, probability)
    })

    spark.createDataFrame(row,structType).createOrReplaceTempView("tb_save")

    spark.sql(
      s"""
         |insert overwrite table persona.yylive_baoxiang_predict_info partition(dt='${dt}', dtype='${dtype}')
         |	select * from tb_save
       """.stripMargin)

  }

}
