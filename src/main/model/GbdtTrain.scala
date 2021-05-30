package model

import org.apache.spark.ml.{Pipeline, PipelineStage, linalg}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import utils.TimeUtils

import scala.collection.mutable.ListBuffer

object GbdtTrain {

  val category_col = Array("sex","bd_consum", "bd_marriage","bd_subconsum", "sys", "start_period_d7", "start_period_d14", "start_period_d30", "city_level", "sjp")

  val numeric_Col = Array("start_cnt_d1","start_cnt_d3","start_cnt_d7","start_cnt_d14","start_cnt_d30",
    "active_days_d1", "active_days_d3", "active_days_d7", "active_days_d14", "active_days_d30",
    "total_watch_dr_d1", "total_watch_dr_d3", "total_watch_dr_d7","total_watch_dr_d14","total_watch_dr_d30",
    "avg_watch_dr_d1", "avg_watch_dr_d3", "avg_watch_dr_d7", "avg_watch_dr_d14","avg_watch_dr_d30",
    "search_cnt_d1", "search_cnt_d3", "search_cnt_d7", "search_cnt_d14", "search_cnt_d30",
    "consume_cnt_d1", "consume_cnt_d3", "consume_cnt_d7","consume_cnt_d14","consume_cnt_d30",
    "channel_msg_cnt_d1","channel_msg_cnt_d3","channel_msg_cnt_d7","channel_msg_cnt_d14","channel_msg_cnt_d30",
    "consume_money_d1","consume_money_d3", "consume_money_d7", "consume_money_d14","consume_money_d30",
    "cancel_cnt_d1","cancel_cnt_d3","cancel_cnt_d7","cancel_cnt_d14","cancel_cnt_d30",
    "subscribe_cnt_d1", "subscribe_cnt_d3","subscribe_cnt_d7","subscribe_cnt_d14","subscribe_cnt_d30",
    "exposure_cnt_d1","exposure_cnt_d3","exposure_cnt_d7","exposure_cnt_d14", "exposure_cnt_d30",
    "click_cnt_d1","click_cnt_d3","click_cnt_d7","click_cnt_d14","click_cnt_d30",
    "push_click_cnt_d1","push_click_cnt_d3", "push_click_cnt_d7","push_click_cnt_d14","push_click_cnt_d30",
    "push_click_day_d1", "push_click_day_d3", "push_click_day_d7","push_click_day_d14","push_click_day_d30", "live_cnt"
  )


  def main(args: Array[String]): Unit = {
    val dt = args(0)
    val dt1 = TimeUtils.changFormat(dt)
    val sparkSession = SparkSession.builder().enableHiveSupport().getOrCreate()
    sparkSession.sparkContext.setLogLevel("warn")

//    val map = getBizMap(sparkSession, dt1)
//    registUDF(sparkSession,map)
    val data = layerSampleData(sparkSession, dt)
    data.show(10, false)
    println("************************+++++++++++++++*************************************")

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
      .setMaxIter(20)

    println("getMaxIter:" + trainer.getMaxIter)

/*    val layers = Array[Int](pca.getK, 280, 140, 2)
    println("layers 0:" + layers(0))
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(50)
      .setFeaturesCol("pcaFeatures")
      .setLabelCol("label")
      .setSolver("l-bfgs")*/

    stagesArray.append(assembler)
    stagesArray.append(pca)
    stagesArray.append(trainer)

    // Chain indexers and GBT in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(stagesArray.toArray)

    // Train model. This also runs the indexers.
    val model = pipeline.fit(data)

    val output = "hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang/pipe"
    model.save(output + "/pipeline_" + dt )

    println("model.stages lens:" + model.stages.length)

    // Make predictions.
    val predictTrain = model.transform(data)
    predictTrain.show(10, false)
    predictTrain.select("label", "prediction")
      .createOrReplaceTempView("gbdt_trained")
    getIndicators(sparkSession, "gbdt_trained")

    val testData = getTestData(sparkSession, dt)
    val predictTest = model.transform(testData)
    predictTest.select("label", "prediction")
      .createOrReplaceTempView("gbdt_test")
    getIndicators(sparkSession, "gbdt_test")

    saveData2hive(sparkSession, dt, predictTest)

    val evaluator = new BinaryClassificationEvaluator()
    evaluator.setMetricName("areaUnderROC")
    val trainAuc= evaluator.evaluate(predictTrain)
    println(" train auc:" + trainAuc)

    val testAuc= evaluator.evaluate(predictTest)
    println(" test auc:" + testAuc)

    val gbtModel = model.stages(13).asInstanceOf[GBTClassificationModel]
//    println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")

    val importances: linalg.Vector = gbtModel.featureImportances
    println("feature importances:" + importances)
    sparkSession.close()
  }

  def saveData2hive(spark:SparkSession, dt: String, dataFrame: DataFrame): Unit ={
    val structFields = Array(
      StructField("hdid",StringType,true),
      StructField("prediction",DoubleType,true),
      StructField("label",IntegerType,true),
      StructField("probability",DoubleType,true)
    )
    val structType = DataTypes.createStructType(structFields)
    val row: RDD[Row] = dataFrame.select("hdid", "prediction", "label", "probability").rdd.map(p => {
      val hdid = p.getString(0)
      val prediction = p.getDouble(1)
      val label = p.getInt(2)
      val probability = p.getAs[DenseVector](3)(1)
      Row(hdid, prediction, label, probability)
    })

    spark.createDataFrame(row,structType).createOrReplaceTempView("tb_save")

    val dt30 = TimeUtils.addDate(dt, 30)
    spark.sql(
      s"""
         |insert overwrite table persona.yylive_dws_user_life_predict partition(dt='${dt30}')
         |	select * from tb_save
       """.stripMargin)
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
    println("pos_data", pos_data.count())
    println("neg_data", neg_data.count())
    val dataFrame = pos_data.union(neg_data.sample(false, ratio * 2))
    dataFrame
  }

  def layerSampleData(sparkSession: SparkSession, dt: String):DataFrame = {
      val dt1 = TimeUtils.changFormat(dt)
      val dt3 = TimeUtils.addDate(dt, -2)
      val dt7 = TimeUtils.addDate(dt, -6)
      val dt9 = TimeUtils.addDate(dt, -8)
      val sqlText = s"""
                       |SELECT * FROM
                       |persona.yylive_dws_user_lifecycle_feature where dt in( '${dt1}', '${dt3}','${dt7}', '${dt9}')""".stripMargin
      println("layerSampleData:" + sqlText)
      val allData = sparkSession.sql(sqlText)
      val data = allData.sample(false, 0.5)
      println("pos_num:" + data.where("label = 1").count())
      println("neg_num:" + data.where("label = 0").count())
      data
  }

  def getTestData(sparkSession: SparkSession, dt: String): DataFrame = {
    val dt30 = TimeUtils.addDate(dt, 30)
    val sqlText = s"""
                     |SELECT * FROM
                     |persona.yylive_dws_user_lifecycle_feature where dt in( '${dt30}')""".stripMargin
    println("getTestData:" + sqlText)
    sparkSession.sql(sqlText)
  }

  def registUDF(sparkSession: SparkSession, map: collection.Map[String, Long]): Unit ={
    sparkSession.udf.register("to_vector", (s: String) =>{
      val bizs: Array[String] = s.split(",")
      if(bizs(0).trim().nonEmpty){
        val index = new Array[Int](bizs.length)
        val index2 = new Array[Double](bizs.length)
        for (i <- bizs.indices) {
          val l: Long = map.getOrElse(bizs(i), 0)
          index(i) = l.toInt
          index2(i) = 1.0
        }
        Vectors.sparse(map.size, index.sortWith(_<_), index2)
      }else{
        val index = new Array[Int](map.size)
        val index2 = new Array[Double](map.size)
        for(i <- 0 until map.size){
          index(i) = i
          index2(i) = 0.0
        }
        Vectors.sparse(map.size, index.sortWith(_<_), index2)
      }
    })
  }

  def getBizMap(sparkSession: SparkSession, dt1: String): collection.Map[String, Long] ={
    val dataFrame = sparkSession.sql(
      s"""
         |select bizs from persona.yylive_dws_user_lifecycle_feature  lateral view explode(split(biz_watch_top5_d30,",")) a as bizs
         | WHERE dt="${dt1}" and biz_watch_top5_d30 != '' and biz_watch_top5_d30 is not null group by bizs
       """.stripMargin)
    dataFrame.rdd.map(p => {
      p.getString(0)
    }).distinct().zipWithIndex().collectAsMap()
  }

}
