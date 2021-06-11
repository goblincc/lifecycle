package analysis

import analysis.gbtTrain.getIndicators
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.sql.{DataFrame, SparkSession}

object gbt2 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .config("spark.hadoop.validateOutputSpecs", value = false)
      .enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("warn")

    val data = spark.sql(
      s"""
         select
         |if(talk_dr > 137, 1, 0) as labels,
         |hdid,
         |bd_sex,
         |bd_age,
         |bd_consume,
         |bd_marriage,
         |bd_high_value,
         |bd_low_value,
         |bd_sing,
         |bd_dance,
         |bd_talk,
         |bd_outdoor,
         |bd_game,
         |bd_sport
         |from persona.spring_activity_baidu_userprofile_analyse
         |where bd_sex is not NULL
         |and bd_age is not NULL
         |and bd_consume is not NULL
         |and bd_marriage is not NULL
         |and bd_high_value is not NULL
         |and bd_low_value is not NULL
         |and bd_sing is not NULL
         |and bd_dance is not NULL
         |and bd_talk is not NULL
         |and bd_outdoor is not NULL
         |and bd_game is not NULL
         |and bd_sport is not NULL
       """.stripMargin)
    data.show(20, false)

//    println("pos:"+ data.where("label = 1").count())
//    println("neg:"+ data.where("label = 0").count())

    val formula =
      s"""
         |labels ~ bd_sex + bd_age + bd_consume + bd_marriage + bd_high_value + bd_low_value + bd_sing +
         |bd_dance + bd_talk + bd_outdoor + bd_game + bd_sport
       """.stripMargin
    val rformula = new RFormula()
      .setFormula(formula)
      .setFeaturesCol("features")
      .setLabelCol("label")
    val output = rformula.fit(data).transform(data)
    println("rformula output:")
    output.show(false)

    val trainer = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")

    val splits = output.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = sampleData(splits(0))
    println("train_pos:"+ training.where("label = 1.0").count())
    println("train_neg:"+ training.where("label = 0.0").count())
    val test = splits(1)
    println("test_pos:"+ test.where("label = 1.0").count())
    println("test_neg:"+ test.where("label = 0.0").count())


    val model: GBTClassificationModel = trainer.fit(training)

    val traindf = model.transform(training)
    traindf.show(30,false)
    traindf.select("label", "prediction")
      .createOrReplaceTempView("train")
    getIndicators(spark, "train")

    val testdf = model.transform(test)
    testdf.show(30,false)
    testdf.select("label", "prediction")
      .createOrReplaceTempView("test")
    getIndicators(spark, "test")

    val evaluator = new BinaryClassificationEvaluator()
    evaluator.setMetricName("areaUnderROC")
    val testAuc= evaluator.evaluate(testdf)
    println(" test auc:" + testAuc)

    println("importance:"+ model.featureImportances)
  }

  def getIndicators(sparkSession: SparkSession, table: String): Unit ={
    val matrix = sparkSession.sql(
      s"""
        select
            '${table}' as type
            ,predict_cnt
            ,(TP + FN) as real_cnt
            ,(TP + FP) as predict_real_cnt
            ,(TP + TN)/ predict_cnt as accuarcy
            ,TP/(TP + FP) as precise
            ,TP/(TP + FN) as recall
            ,TP
            ,FP
            ,TN
            ,FN
        from (
            select
                count(1) as predict_cnt
                ,count(if(label = 1.0 and prediction= 1.0, 1, null)) as TP
                ,count(if(label = 0.0 and prediction= 1.0, 1, null)) as FP
                ,count(if(label = 0.0 and prediction= 0.0, 1, null)) as TN
                ,count(if(label = 1.0 and prediction= 0.0, 1, null)) as FN
            from ${table}
        )b
        """
    )
    matrix.show()
  }

  def sampleData(data: DataFrame): DataFrame ={
    val pos_data = data.where("label = 1.0")
    val neg_data = data.where("label = 0.0")
    val ratio = pos_data.count() * 1.0/neg_data.count()
    println("pos_data", pos_data.count())
    println("neg_data", neg_data.count())
    val dataFrame = pos_data.union(neg_data.sample(false, ratio * 3))
    dataFrame
  }
}
