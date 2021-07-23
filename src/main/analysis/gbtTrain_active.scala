package analysis


import utils.ModelUtils.sampleData
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, SparkSession}
import utils.ModelUtils.getIndicators
import utils.TimeUtils

import scala.collection.mutable.ListBuffer

object gbtTrain_active {
  def main(args: Array[String]): Unit = {
    val dts = args(0)
    val dt = TimeUtils.changFormat(dts)
    val spark = SparkSession.builder()
      .config("spark.hadoop.validateOutputSpecs", value = false)
      .enableHiveSupport().getOrCreate()
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
         |label,
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
         |cast(is_exposure_v2 as int) as is_exposure_v2,
         |if(age is null, 'other', age) as yy_age,
         |dt,
         |row_number() over (partition BY hdid ORDER BY is_exposure_v2 desc, dt asc ) AS rank
         |from persona.yylive_baoxiang_feature_day_v2  where dt >= '2021-05-13' and dt <= '2021-06-01'
         |and last_active_dt <= date_add(dt, -3)) as a
         |left join (
         |SELECT hdid,
         |       event_act_list_d4_d7,
         |       event_act_list_d8_d14,
         |       event_act_list_d15_d30,
         |       dt
         |FROM persona.yylive_dws_web_event_act_d
         |WHERE dt >= '2021-05-13' and dt <= '2021-06-01') as b
         |on a.hdid = b.hdid and a.dt = b.dt where rank = 1
       """.stripMargin
    val datas = spark.sql(sqlTxt).repartition(500).cache()
    val data = sampleDatas(datas, spark)

    val num_feature = Array("start_cnt_d7","start_cnt_d14","start_cnt_d30", "active_days_d7",
      "active_days_d14","active_days_d30", "total_watch_dr_d7", "total_watch_dr_d14","total_watch_dr_d30",
      "consume_cnt_d7","consume_cnt_d14", "consume_cnt_d30","push_click_cnt_d7",
      "push_click_cnt_d14", "push_click_cnt_d30")

    val stagesArray = new ListBuffer[PipelineStage]()

    val formula =
      s"""
         |label ~ sex + yy_age + city_level + sjp + sys + bd_consum + bd_marriage + bd_subconsum + bd_age
       """.stripMargin

    val rformula = new RFormula()
      .setFormula(formula)
      .setFeaturesCol("catVec")
      .setLabelCol("label")
      .setHandleInvalid("skip")

    stagesArray.append(rformula)

    tf_idf(stagesArray, "applists", 10)
    tf_idf(stagesArray, "event_act_list_d4_d7", 10)
    tf_idf(stagesArray, "event_act_list_d8_d14", 12)
    tf_idf(stagesArray, "event_act_list_d15_d30", 14)

    val assembler = new VectorAssembler()
      .setInputCols(Array("catVec", "applists_vec") ++ num_feature)
      .setOutputCol("assemble")

    stagesArray.append(assembler)

    val pca = new PCA()
      .setInputCol("assemble")
      .setOutputCol("pcaFeatures")
      .setK(20)
    stagesArray.append(pca)

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("is_exposure_v2"))
      .setOutputCols(Array("is_exposure_v2_vec"))
    stagesArray.append(encoder)

    val assemblerAll = new VectorAssembler()
      .setInputCols(Array("pcaFeatures", "is_exposure_v2_vec"))
      .setOutputCol("assembleAll")

    stagesArray.append(assemblerAll)

    val trainer = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("assembleAll")
      .setMaxDepth(3)
      .setMaxIter(20)

    stagesArray.append(trainer)

    val pipeline = new Pipeline()
      .setStages(stagesArray.toArray)

    val model = pipeline.fit(data)

    val output = "hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang/pipe"
    model.write.overwrite().save(output + "/pipelineActive_" + dt )

    val predictTrain = model.transform(data).cache()
    predictTrain.show(10, false)
    predictTrain.select("label", "prediction")
      .createOrReplaceTempView("trained")
    getIndicators(spark, "trained")

    val evaluator = new BinaryClassificationEvaluator()
    evaluator.setMetricName("areaUnderROC")
    val trainAuc= evaluator.evaluate(predictTrain)
    println(" train auc:" + trainAuc)

    val sql = s"""
                 |select a.*,
                 |if(event_act_list_d4_d7 is null, '', event_act_list_d4_d7 ) as event_act_list_d4_d7,
                 |if(event_act_list_d8_d14 is null, '', event_act_list_d8_d14) as event_act_list_d8_d14,
                 |if(event_act_list_d15_d30 is null, '', event_act_list_d15_d30) as event_act_list_d15_d30
                 |from (
                 |select
                 |hdid,
                 |label,
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
                 |cast(is_exposure_v2 as int) as is_exposure_v2,
                 |if(applist is null, '', applist) as applists,
                 |if(age is null, 'other', age) as yy_age,
                 |dt
                 |from persona.yylive_baoxiang_feature_day_v2 where dt = '2021-06-03') as a
                 |left join (
                 |SELECT hdid,
                 |       event_act_list_d4_d7,
                 |       event_act_list_d8_d14,
                 |       event_act_list_d15_d30,
                 |       dt
                 |FROM persona.yylive_dws_web_event_act_d
                 |WHERE dt = '2021-06-03') as b
                 |on a.hdid = b.hdid and a.dt = b.dt
       """.stripMargin
    println("test_sql:", sql)
    val testData = spark.sql(sql)

    val predictTest = model.transform(testData)
    predictTest.select("label", "prediction")
      .createOrReplaceTempView("test")
    getIndicators(spark, "test")

    val testAuc= evaluator.evaluate(predictTest)
    println(" test auc:" + testAuc)
    spark.close()

  }

  def sampleDatas(data: DataFrame, spark: SparkSession): DataFrame ={
    val pos_data = data.where("is_exposure_v2 = 1 and label = 1")
    val pos_data_2 = data.where("is_exposure_v2 = 0 and label = 1")
//    val ratio1 = pos_data.count() * 1.0/pos_data_2.count()
    val ratio1 = 0.0678
    println("ratio1", ratio1)

    val pos_data_all = pos_data.union(pos_data_2.sample(false, ratio1 * 0.5))

    val neg_data = data.where("is_exposure_v2 = 1 and label = 0")
    val neg_data_2 = data.where("is_exposure_v2 = 0 and label = 0")
//    val ratio2 = neg_data.count() * 1.0/neg_data_2.count()
    val ratio2 = 0.0166
    println("ratio2", ratio2)

    val neg_data_all = neg_data.sample(false, 0.3).union(neg_data_2.sample(false, ratio2 * 0.25))

    val dataFrame = pos_data_all.union(neg_data_all)
    dataFrame.createOrReplaceTempView("sample")
    spark.sql("select label, is_exposure_v2, count(*) as cnt from sample group by label, is_exposure_v2").show(false)

    dataFrame
  }


  def tf_idf(stagesArray:ListBuffer[PipelineStage], input: String, numFeatures: Int): Unit ={
    val tokenizer = new RegexTokenizer()
      .setInputCol(input)
      .setOutputCol(input + "_token")
      .setPattern("\\|")
    stagesArray.append(tokenizer)

    val hashingTF = new HashingTF()
      .setInputCol(input + "_token").setOutputCol(input + "_tf").setNumFeatures(numFeatures)
    stagesArray.append(hashingTF)

    val idf = new IDF().setInputCol(input + "_tf").setOutputCol(input + "_vec")
    stagesArray.append(idf)

  }
}
