package model

import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{GBTClassifier, LogisticRegression}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.sql.{DataFrame, SparkSession}
import utils.TimeUtils

import scala.collection.mutable.ListBuffer

object DocCTR {
  val doc2vecPath = "hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang"

  val category_col = Array("sex","bd_consum", "bd_marriage","bd_subconsum", "sys", "start_period_d7", "start_period_d14",
  "start_period_d30", "city_level", "sjp","tag_type")

  val numeric_Col = Array("start_cnt_d1","start_cnt_d3","start_cnt_d7","start_cnt_d14","start_cnt_d30",
    "active_days_d1", "active_days_d3", "active_days_d7", "active_days_d14", "active_days_d30",
    "total_watch_dr_d1", "total_watch_dr_d3", "total_watch_dr_d7","total_watch_dr_d14","total_watch_dr_d30",
    "avg_watch_dr_d1", "avg_watch_dr_d3", "avg_watch_dr_d7", "avg_watch_dr_d14","avg_watch_dr_d30",
    "consume_cnt_d1", "consume_cnt_d3", "consume_cnt_d7","consume_cnt_d14","consume_cnt_d30",
    "exposure_cnt_d1","exposure_cnt_d3","exposure_cnt_d7","exposure_cnt_d14", "exposure_cnt_d30",
    "click_cnt_d1","click_cnt_d3","click_cnt_d7","click_cnt_d14","click_cnt_d30",
    "push_click_cnt_d1","push_click_cnt_d3", "push_click_cnt_d7","push_click_cnt_d14","push_click_cnt_d30",
    "push_click_day_d1", "push_click_day_d3", "push_click_day_d7","push_click_day_d14","push_click_day_d30",
    "title_length", "content_length", "show_1","show_3","show_7","show_14","show_30","show_90","click_1","click_3","click_7",
    "click_14","click_30","click_90","wilson_ctr_1","wilson_ctr_3","wilson_ctr_7","wilson_ctr_14","wilson_ctr_30" )

  def main(args: Array[String]): Unit = {
    val dts = args(0)
    val dt = TimeUtils.changFormat(dts)
    val spark = SparkSession.builder()
      .config("spark.hadoop.validateOutputSpecs", value = false)
      .enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("warn")
    val dt7 = TimeUtils.addDate(dts, -6)
    val dt14 = TimeUtils.addDate(dts, -13)
    val dt21 = TimeUtils.addDate(dts, -20)
    val dt28 = TimeUtils.addDate(dts, -27)
    val sqltxt =
      s"""
         |select * from persona.yylive_dws_user_docid_ctr_feature  WHERE  dt in('${dt28}' ,'${dt21}','${dt14}','${dt7}', '${dt}') and title_length is not null
       """.stripMargin
    println("train sql:",sqltxt)
    val datas = spark.sql(sqltxt)
    datas.show(5, false)
    val data = sampleData(datas)

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

    val assemblerInputs = category_col.map(_ + "Vec") ++ numeric_Col

    val assembler = new VectorAssembler()
      .setInputCols(assemblerInputs)
      .setOutputCol("features")

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(70)
    println("pca len:" + pca.getK)

    val trainer = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("pcaFeatures")
      .setMaxDepth(6)
      .setMaxIter(20)

    stagesArray.append(assembler)
    stagesArray.append(pca)
    stagesArray.append(trainer)

    val pipeline = new Pipeline()
      .setStages(stagesArray.toArray)

    // Train model. This also runs the indexers.
    val model = pipeline.fit(data)

    val output = "hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang/pipe"
    model.write.overwrite().save(output + "/pipeDocCTR_" + dt)

    val predictTrain = model.transform(data)
    predictTrain.show(10, false)
    predictTrain.select("label", "prediction")
      .createOrReplaceTempView("trained")
    getIndicators(spark, "trained")

    val evaluator = new BinaryClassificationEvaluator()
    evaluator.setMetricName("areaUnderROC")
    val trainAuc= evaluator.evaluate(predictTrain)
    println(" train auc:" + trainAuc)

    val test_dt = TimeUtils.addDate(dts, -10)
    val testData = spark.sql(
      s"""
         |select *  from persona.yylive_dws_user_docid_ctr_feature  WHERE dt = '${test_dt}' and title_length is not null
       """.stripMargin)

    val predictTest = model.transform(testData)
    predictTest.select("label", "prediction")
      .createOrReplaceTempView("test")
    getIndicators(spark, "test")

    val testAuc= evaluator.evaluate(predictTest)
    println(" test auc:" + testAuc)

    spark.close()
  }


  def getDocVec(): collection.Map[String, DenseVector] ={
    val w2vModel: Word2VecModel = Word2VecModel.read.load(doc2vecPath + "/model")
    w2vModel.getVectors.rdd.map(p => {
      (p.getString(0), p.getAs[DenseVector](1))
    }).collectAsMap()
  }

  def getIndicators(sparkSession: SparkSession, table: String): Unit ={
    val matrix = sparkSession.sql(
      s"""
        select
            '${table}' as type
            ,predict_cnt
            ,(TP + FN) as real_loss_cnt
            ,(TP + FP) as predict_loss_cnt
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
    val pos_data = data.where("label = 1")
    val neg_data = data.where("label = 0")
    val ratio = pos_data.count() * 1.0/neg_data.count()
    println("ratio", ratio)
    val dataFrame = pos_data.union(neg_data.sample(false, ratio * 20))
    println("pos_data",dataFrame.where("label = 1").count())
    println("neg_data",dataFrame.where("label = 0").count())
    dataFrame

  }




}
