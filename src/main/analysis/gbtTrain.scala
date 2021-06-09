package analysis

import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier, LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer

object gbtTrain {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .config("spark.hadoop.validateOutputSpecs", value = false)
      .enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("warn")

    val data = spark.sql(
      s"""
         select
         |if(sing_dr > 300, 1, 0) as label,
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
    val splits = data.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val category_col = Array("bd_sex","bd_age","bd_consume", "bd_marriage", "bd_high_value", "bd_low_value","bd_sing",
      "bd_dance", "bd_talk", "bd_outdoor",  "bd_game", "bd_sport"
    )
    //    val formula =
    //      s"""
    //         |sing_label ~ bd_sex + bd_age + bd_consume + bd_marriage + bd_high_value + bd_low_value + bd_sing +
    //         |bd_dance + bd_talk + bd_outdoor + bd_game + bd_sport
    //       """.stripMargin
    //
    //    val rformula = new RFormula()
    //      .setFormula(formula)
    //      .setFeaturesCol("features")
    //      .setLabelCol("label")
    //
    //    val output = rformula.fit(data).transform(data)
    //
    //    output.show(false)

    val stagesArray = new ListBuffer[PipelineStage]()
    val indexArray = new ListBuffer[String]()
    val vecArray = new ListBuffer[String]()
    for(cate <- category_col){
      val indexer = new StringIndexer().setInputCol(cate).setOutputCol(s"${cate}Index")
      indexArray.append(s"${cate}Index")
      vecArray.append(s"${cate}Vec")
      stagesArray.append(indexer)
    }

    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(indexArray.toArray)
      .setOutputCols(vecArray.toArray)

    stagesArray.append(oneHotEncoder)

    val assemblerInputs = category_col.map(_ + "Vec")

    val assembler = new VectorAssembler()
      .setInputCols(assemblerInputs)
      .setOutputCol("features")

    val trainer = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")

    stagesArray.append(assembler)
    stagesArray.append(trainer)


    val pipeline = new Pipeline()
      .setStages(stagesArray.toArray)

    // Train model. This also runs the indexers.
    val model = pipeline.fit(training)

    val predict = model.transform(test)
    predict.show(30,false)
    predict.select("label", "prediction")
      .createOrReplaceTempView("test")
    getIndicators(spark, "test")

    val evaluator = new BinaryClassificationEvaluator()
    evaluator.setMetricName("areaUnderROC")
    val trainAuc= evaluator.evaluate(predict)
    println(" test auc:" + trainAuc)

    val stage = model.stages
    print("length:", stage.length)

    val gbtModel = stage(14).asInstanceOf[GBTClassificationModel]
    val importances = gbtModel.featureImportances
    println("importances:"+ importances)

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
}
