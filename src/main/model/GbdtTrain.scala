package model

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession
import utils.TimeUtils

object GbdtTrain {
  def main(args: Array[String]): Unit = {
    val dt = args(0)
    val dt1 = TimeUtils.changFormat(dt)
    val dt7 = TimeUtils.addDate(dt, 7)
    val sparkSession = SparkSession.builder().enableHiveSupport().getOrCreate()

    val sqltext =
      s"""
         |select
         | hdid ,
         |  if(sex is null or sex == 2, 1, sex) as sex ,
         |	nvl(bd_consum,'other') as bd_consum ,
         |	nvl(bd_marriage , 'other') as bd_marriage,
         |	nvl(bd_subconsum , 'other') as bd_subconsum,
         |	nvl(sys, 2) as sys,
         |	start_cnt_d1 ,
         |	start_cnt_d3 ,
         |	start_cnt_d7 ,
         |	start_cnt_d14 ,
         |	start_cnt_d30 ,
         |	nvl(start_period_d7, 20) as start_period_d7,
         |	nvl(start_period_d14, 20) as start_period_d14,
         |	nvl(start_period_d30, 20) as  start_period_d30,
         |	active_days_d1 ,
         |	active_days_d3 ,
         |	active_days_d7 ,
         |	active_days_d14 ,
         |	active_days_d30 ,
         |	total_watch_dr_d1 ,
         |	total_watch_dr_d3 ,
         |	total_watch_dr_d7 ,
         |	total_watch_dr_d14 ,
         |	total_watch_dr_d30 ,
         |	avg_watch_dr_d1 ,
         |	avg_watch_dr_d3 ,
         |	avg_watch_dr_d7 ,
         |	avg_watch_dr_d14 ,
         |	avg_watch_dr_d30 ,
         |	biz_watch_top5_d30 ,
         |	search_cnt_d1 ,
         |	search_cnt_d3 ,
         |	search_cnt_d7 ,
         |	search_cnt_d14 ,
         |	search_cnt_d30 ,
         |	consume_cnt_d1 ,
         |	consume_cnt_d3 ,
         |	consume_cnt_d7 ,
         |	consume_cnt_d14 ,
         |	consume_cnt_d30 ,
         |	consume_money_d1 ,
         |	consume_money_d3 ,
         |	consume_money_d7 ,
         |	consume_money_d14 ,
         |	consume_money_d30 ,
         |	channel_msg_cnt_d1 ,
         |	channel_msg_cnt_d3 ,
         |	channel_msg_cnt_d7 ,
         |	channel_msg_cnt_d14 ,
         |	channel_msg_cnt_d30 ,
         |	cancel_cnt_d1 ,
         |	cancel_cnt_d3 ,
         |	cancel_cnt_d7 ,
         |	cancel_cnt_d14 ,
         |	cancel_cnt_d30 ,
         |	subscribe_cnt_d1 ,
         |	subscribe_cnt_d3 ,
         |	subscribe_cnt_d7 ,
         |	subscribe_cnt_d14 ,
         |	subscribe_cnt_d30 ,
         |	exposure_cnt_d1 ,
         |	exposure_cnt_d3 ,
         |	exposure_cnt_d7 ,
         |	exposure_cnt_d14 ,
         |	exposure_cnt_d30 ,
         |	click_cnt_d1 ,
         |	click_cnt_d3 ,
         |	click_cnt_d7 ,
         |	click_cnt_d14 ,
         |	click_cnt_d30 ,
         |	push_click_cnt_d1  ,
         |	push_click_cnt_d3  ,
         |	push_click_cnt_d7  ,
         |	push_click_cnt_d14  ,
         |	push_click_cnt_d30  ,
         |	push_click_day_d1  ,
         |	push_click_day_d3  ,
         |	push_click_day_d7  ,
         |	push_click_day_d14  ,
         |	push_click_day_d30 ,
         |if(b.hdid is null, 0, 1) as label from (
         |select * from persona.yylive_dws_user_feature where dt = '${dt1}' and active_days_d30 >= 2) as a
         |left join (
         |select hdid from persona.yylive_ods_idinfo_day where dt = '${dt7}' group by hdid) as b on a.hdid = b.hdid
         |left join (select hdid from persona.yylive_dws_metrics_install_total  WHERE dt="2021-05-07" and dt = '${dt1}') as c
         | on a.hdid = c.hdid where c.hdid is null
       """.stripMargin

    val data = sparkSession.sql(sqltext)



    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a GBT model.
    val gbt = new GBTClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)


    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")


    // Chain indexers and GBT in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

  }
}
