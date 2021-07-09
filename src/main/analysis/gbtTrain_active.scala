package analysis


import utils.ModelUtils.sampleData
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, SparkSession}
import utils.ModelUtils.getIndicators

import scala.collection.mutable.ListBuffer

object gbtTrain_active {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .config("spark.hadoop.validateOutputSpecs", value = false)
      .enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("warn")

    val sqlTxt =
      s"""
         |select *,
         |if(applist is null, '', applist) as applists,
         |if(age is null, 'other', age) as yy_age
         |from persona.yylive_baoxiang_feature_day where dt >= '2021-05-13' and dt <= '2021-06-01' and is_current_active = 1 and last_active_dt >= date_add(dt, -3)
       """.stripMargin
    val datas = spark.sql(sqlTxt)
    val data = sampleDatas(datas)
    val num_feature = Array("start_cnt_d1","start_cnt_d3", "start_cnt_d7","start_cnt_d14","start_cnt_d30", "active_days_d1", "active_days_d3", "active_days_d7",
      "active_days_d14","active_days_d30", "total_watch_dr_d1", "total_watch_dr_d3", "total_watch_dr_d7", "total_watch_dr_d14","total_watch_dr_d30",
      "consume_cnt_d1","consume_cnt_d3", "consume_cnt_d7","consume_cnt_d14", "consume_cnt_d30", "push_click_cnt_d1","push_click_cnt_d3","push_click_cnt_d7",
      "push_click_cnt_d14", "push_click_cnt_d30")

    val stagesArray = new ListBuffer[PipelineStage]()

    val formula =
      s"""
         |label ~ sex + yy_age + city_level + sjp + sys + is_exposure_v2
       """.stripMargin

    val rformula = new RFormula()
      .setFormula(formula)
      .setFeaturesCol("catVec")
      .setLabelCol("label")
      .setHandleInvalid("skip")

    stagesArray.append(rformula)

    val tokenizer = new RegexTokenizer()
      .setInputCol("applists")
      .setOutputCol("words")
      .setPattern("\\|")
    stagesArray.append(tokenizer)

    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(10)
    stagesArray.append(hashingTF)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("appVec")
    stagesArray.append(idf)

    val assembler = new VectorAssembler()
      .setInputCols(Array("catVec", "appVec") ++ num_feature)
      .setOutputCol("assemble")

    stagesArray.append(assembler)

    val trainer = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("assemble")
      .setMaxDepth(6)
      .setMaxIter(20)

    stagesArray.append(trainer)

    val pipeline = new Pipeline()
      .setStages(stagesArray.toArray)

    val model = pipeline.fit(data)

    val output = "hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang/pipe"
    model.write.overwrite().save(output + "/pipelineActive_2021-06-15" )

    val predictTrain = model.transform(data)
    predictTrain.show(10, false)
    predictTrain.select("label", "prediction")
      .createOrReplaceTempView("trained")
    getIndicators(spark, "trained")

    val evaluator = new BinaryClassificationEvaluator()
    evaluator.setMetricName("areaUnderROC")
    val trainAuc= evaluator.evaluate(predictTrain)
    println(" train auc:" + trainAuc)

    val sql = s"""
                 |select *,
                 |if(applist is null, '', applist) as applists,
                 |if(age is null, 'other', age) as yy_age
                 |from persona.yylive_baoxiang_feature_day where dt = '2021-06-03'
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

  def sampleDatas(data: DataFrame): DataFrame ={
    val pos_data = data.where("is_exposure_v2 = 1 and label = 1")
    val pos_data_2 = data.where("is_exposure_v2 = 0 and label = 1")
    val ratio = pos_data.count() * 1.0/pos_data_2.count()
    println("ratio", ratio)

    val pos_data_all = pos_data.union(pos_data_2.sample(false, ratio))
    val neg_data = data.where("label = 0")
    val ratio2 = pos_data_all.count() * 1.0/neg_data.count()
    println("ratio2", ratio2)

    val dataFrame = pos_data_all.union(neg_data.sample(false, ratio2 * 10))
    println("pos_data",dataFrame.where("label = 1").count())
    println("neg_data",dataFrame.where("label = 0").count())
    dataFrame
  }

}
