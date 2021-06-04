package model

import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.ml.{Pipeline, PipelineStage, linalg}
import org.apache.spark.sql.{DataFrame, SparkSession}
import utils.TimeUtils

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

object DocVecCTR {
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
    val docVecMap: collection.Map[String, DenseVector] = getDocVec()

    val doc2Int: collection.Map[String, Int] = spark.sql(s"""SELECT * FROM persona.yylive_dws_doc_index  WHERE dt='${dt}'""")
      .rdd.map(p => {
      (p.getAs[String](0), p.getAs[Int](1))
    }).collectAsMap()

    val alsModel: ALSModel = ALSModel.read.load("hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang/ALS/AlsModel_" + dt)

    val intToVector: collection.Map[Int, linalg.Vector] = alsModel.itemFactors.rdd.map(p => {
      val idx = p.getAs[Int](0)
      val arr = p.getAs[mutable.WrappedArray[Float]](1)
      val vector: linalg.Vector = Vectors.dense(
        arr.map(_.toDouble).toArray
      )
      (idx, vector)
    }).collectAsMap()

    registUDF(spark, docVecMap,doc2Int,intToVector)
    println("*******************************************************")
    val dt7 = TimeUtils.addDate(dts, -6)
    val dt14 = TimeUtils.addDate(dts, -13)
    val dt21 = TimeUtils.addDate(dts, -20)
    val dt28 = TimeUtils.addDate(dts, -27)
    val sqltxt =
      s"""
         |select *, docVec2(doc_id) as docVec from persona.yylive_dws_user_docid_ctr_feature
         |WHERE  dt in('${dt28}' ,'${dt21}','${dt14}','${dt7}', '${dt}') and title_length is not null
       """.stripMargin

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

    val assemblerInputs = category_col.map(_ + "Vec") ++ numeric_Col ++ Array("docVec")

    val assembler = new VectorAssembler()
      .setInputCols(assemblerInputs)
      .setOutputCol("features")
      .setHandleInvalid("skip")

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(70)
    println("pca len:" + pca.getK)


   /* /**
      * setMaxIter，最大迭代次数，训练的截止条件，默认100次
      * setFamily，binomial(二分类)/multinomial(多分类)/auto，默认为auto。设为auto时，会根据schema或者样本中实际的class情况设置是二分类还是多分类，最好明确设置
      * setElasticNetParam，弹性参数，用于调节L1和L2之间的比例，两种正则化比例加起来是1，详见后面正则化的设置，默认为0，只使用L2正则化，设置为1就是只用L1正则化
      * setRegParam，正则化系数，默认为0，不使用正则化
      * setTol，训练的截止条件，两次迭代之间的改善小于tol训练将截止
      * setFitIntercept，是否拟合截距，默认为true
      * setStandardization，是否使用归一化，这里归一化只针对各维特征的方差进行
      * setThresholds/setThreshold，设置多分类/二分类的判决阈值，多分类是一个数组，二分类是double值
      * setAggregationDepth，设置分布式统计时的层数，主要用在treeAggregate中，数据量越大，可适当加大这个值，默认为2
      */
    val trainer = new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setRegParam(0.3)
      .setElasticNetParam(0.8)*/

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
    model.write.overwrite().save(output + "/pipelineCTR_" + dt)

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

    val sql = s"""
                 |select *, docVec2(doc_id) as docVec from persona.yylive_dws_user_docid_ctr_feature  WHERE dt = '${test_dt}' and title_length is not null
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

  def registUDF(spark: SparkSession, map: collection.Map[String, DenseVector], doc2Int: collection.Map[String, Int], intToVector: collection.Map[Int, linalg.Vector]): Unit = {
    spark.udf.register("docVector", (s: String) => {
      map.getOrElse(s, Vectors.dense(Array(0.0, 0.0, 0.0, 0.0, 0.0)))
    })

    spark.udf.register("docVec2", (s: String) => {
      val idx = doc2Int.getOrElse(s, -1)
      intToVector.getOrElse(idx, Vectors.dense(Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    })

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
