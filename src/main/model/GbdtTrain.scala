package model

import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, OneHotEncoderModel, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}
import utils.TimeUtils

import scala.collection.mutable.ListBuffer

object GbdtTrain {
  def main(args: Array[String]): Unit = {
    val dt = args(0)
    val dt1 = TimeUtils.changFormat(dt)
    val dt2 = TimeUtils.addDate(dt, 1)
    val dt7 = TimeUtils.addDate(dt, 7)
    val sparkSession = SparkSession.builder().enableHiveSupport().getOrCreate()
    sparkSession.sparkContext.setLogLevel("warn")
    val map = getBizMap(sparkSession, dt1)
    registUDF(sparkSession,map)

    val sqlText =
      s"""
         |  select
         |  a.hdid ,
         |  if(sex is null or sex == 2, 1, sex) as sex ,
         |	if(bd_consum is null or trim(bd_consum) == '','other', bd_consum) as bd_consum ,
         |	if(bd_marriage is null or trim(bd_marriage) == '', 'other', bd_marriage) as bd_marriage,
         |	if(bd_subconsum is null or trim(bd_subconsum) == '' , 'other', bd_subconsum) as bd_subconsum,
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
         |	to_vector(nvl(biz_watch_top5_d30, "")) as  biz_watch_top5_d30,
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
         |select * from persona.yylive_dws_user_feature where dt = '${dt1}') as a
         |left join (
         |select hdid from persona.yylive_ods_idinfo_day where dt >= '${dt2}' and dt <= '${dt7}' group by hdid) as b on a.hdid = b.hdid
       """.stripMargin
    println("sqlText", sqlText)
    val data = sparkSession.sql(sqlText)
    data.show(5, false)

//    val data = sampleData(datas)

//    类别型特征采用onehot处理
    val category_col = getCategoryCol()
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

    val numeric_Col = getNumericCol()

    val assemblerInputs = category_col.map(_ + "Vec") ++ numeric_Col

    val assembler = new VectorAssembler()
      .setInputCols(assemblerInputs)
      .setOutputCol("features")

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a GBT model.
    val gbt = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(10)

    stagesArray.append(assembler)
    stagesArray.append(gbt)
    // Chain indexers and GBT in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(stagesArray.toArray)

    // Train model. This also runs the indexers.
    val model = pipeline.fit(data)

    // Make predictions.
    val predictions = model.transform(data)
    predictions.show(10, false)

    val train_out = predictions.select("label", "prediction")

    train_out.createOrReplaceTempView("gbdt_trained")
    val trained_matrix = sparkSession.sql(
      s"""
        select
            'trained data' as type
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
            from gbdt_trained
        )b
        """
    )
    trained_matrix.show()
    sparkSession.close()

  }

  def sampleData(data: DataFrame): DataFrame ={
    val pos_data = data.where("label = 1")
    val neg_data = data.where("label = 0")
    val ratio = pos_data.count()/neg_data.count()
    println("pos_data", pos_data.count())
    println("neg_data", neg_data.count())
    val dataFrame = pos_data.union(neg_data.sample(false, ratio * 2))
    dataFrame
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
         |select bizs  from persona.yylive_dws_user_feature  lateral view explode(split(biz_watch_top5_d30,",")) a as bizs
         | WHERE dt="${dt1}" and biz_watch_top5_d30 is not null group by bizs
       """.stripMargin)
    dataFrame.rdd.map(p => {
      p.getString(0)
    }).distinct().zipWithIndex().collectAsMap()
  }

  def getCategoryCol():Array[String]={
    Array("sex","bd_consum", "bd_marriage","bd_subconsum", "sys", "start_period_d7", "start_period_d14", "start_period_d30")
  }

  def getNumericCol():Array[String]={
    Array("start_cnt_d1","start_cnt_d3","start_cnt_d7","start_cnt_d14","start_cnt_d30",
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
      "push_click_day_d1", "push_click_day_d3", "push_click_day_d7","push_click_day_d14","push_click_day_d30"
    )
  }
}
