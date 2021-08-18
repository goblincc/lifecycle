package analysis

import analysis.gbtTrain_active.tf_idf
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.ListBuffer

object gbt_risk {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .config("spark.hadoop.validateOutputSpecs", value = false)
      .enableHiveSupport()
      .getOrCreate()
    spark.sparkContext.setLogLevel("error")

    val df = spark.sql(
      s"""
         |SELECT a.*,
         |    cast(is_nick_modify as string) as is_nick_modifys,
         |    (cnt_30 + cnt_60 + cnt_90) as cnt_90_all,
         |    (day_30 + day_60 + day_90) as day_90_all,
         |    (sid_30 + sid_60 + sid_90) as sid_90_all,
         |    (sid_all_30 + sid_all_60 + sid_all_90) as sid_all_90_dr,
         |    (gift_cnt_30 + gift_cnt_60 + gift_cnt_90) as gift_cnt_90_all,
         |    (sum_30 + sum_60 + sum_90) as sum_90_all,
         |    (alldt_30 + alldt_60 + alldt_90) as alldt_90_all,
         |    (amount_30 + amount_60 + amount_90) as amount_90_all,
         |    nvl(alldt_30/dtcnt_30,0) AS avg_pay_times_30,
         |    nvl(alldt_60/dtcnt_60,0) AS avg_pay_times_60,
         |    nvl(alldt_90/dtcnt_90,0) AS avg_pay_times_90,
         |    nvl(chid_30/alldt_30,0) AS avg_chid_times_30,
         |    nvl(chid_60/alldt_60,0) AS avg_chid_times_60,
         |    nvl(chid_90/alldt_90,0) AS avg_chid_times_90,
         |    nvl(paymethod_30/alldt_30,0) AS avg_method_times_30,
         |    nvl(paymethod_60/alldt_60,0) AS avg_method_times_60,
         |    nvl(paymethod_90/alldt_90,0) AS avg_method_times_90,
         |    nvl(buyerid_cnt/alldt_90,0) AS buyer_pay_ratio,
         |    if(b.uid IS NULL, 0, 1) AS label
         |   FROM
         |     (SELECT *
         |      FROM persona.yylive_uid_feature_info
         |      WHERE dt='2021-08-15'
         |  ) AS a
         |   LEFT JOIN
         |     (SELECT uid
         |      FROM persona.yylive_risk_rule_day_all
         |      WHERE dt = '2021-08-15'
         |      UNION SELECT uid
         |      FROM persona.yylive_ods_blacklist_d
         |      WHERE dt='20210815') AS b ON a.uid = b.uid
       """.stripMargin)


    val splits = df.randomSplit(Array(0.7, 0.3), seed = 11L)

    //对训练集采样
    val training = sampleData(splits(0))
    println("train_pos:"+ training.where("label = 1.0").count())
    println("train_neg:"+ training.where("label = 0.0").count())

    //测试集维持原始比例
    val test = splits(1)
    println("test_pos:"+ test.where("label = 1.0").count())
    println("test_neg:"+ test.where("label = 0.0").count())

    val num_features = Array(
      "hdid_cnt","sid_30","sid_60","sid_90",
      "sid_all_30","sid_all_60","sid_all_90",
      "apporderid_cnt","buyerid_cnt",
      "sub_cnt","cont_d_30","cont_d_60","cont_d_90","cont_all_30",
      "cont_all_60","cont_all_90","avg_cont_30",
      "avg_cont_60","avg_cont_90","gift_cnt_90_all",
      "sum_90_all","stddev_30","stddev_60","stddev_90",
      "avg_30","avg_60","avg_90","dtcnt_30",
      "dtcnt_60","dtcnt_90",
      "alldt_90_all","amount_90_all",
      "avg_amount_30","avg_amount_60","avg_amount_90",
      "stddev_amount_30","stddev_amount_60","stddev_amount_90",
      "chid_30","chid_60","chid_90","max_cnt_30","max_cnt_60","max_cnt_90",
      "avg_cnt_30","avg_cnt_60","avg_cnt_90",
      "avg_amount", "stddev_amount", "max_cnt","avg_cnt","avg_delta_time",
      "stddev_delta_time","avg_all_90","stddev_all_90",
      "max_IP_cnt","avg_IP_cnt","stdev_IP_cnt",
      "avg_pay_times_30", "avg_pay_times_60", "avg_pay_times_90",
      "avg_chid_times_30", "avg_chid_times_60", "avg_chid_times_90",
      "avg_method_times_30", "avg_method_times_60", "avg_method_times_90",
      "buyer_pay_ratio"
    )

    val stagesArray = new ListBuffer[PipelineStage]()

    val formula =
      s"""
         |label ~ is_nick_modifys + has_idnumber
       """.stripMargin

    val rformula = new RFormula()
      .setFormula(formula)
      .setFeaturesCol("catVec")
      .setLabelCol("label")
      .setHandleInvalid("skip")

    stagesArray.append(rformula)

    tf_idf_stage(stagesArray, "events_list_90", 20)
//    tf_idf_stage(stagesArray, "events_list_60", 20)
//    tf_idf_stage(stagesArray, "events_list_30", 20)
    tf_idf_stage(stagesArray, "statuscode_list", 10)
    tf_idf_stage(stagesArray, "appid_list", 10)

    val assembler = new VectorAssembler()
      .setInputCols(num_features ++ Array("catVec", "events_list_90_vec","statuscode_list_vec", "appid_list_vec"))
      .setOutputCol("assemble")
      .setHandleInvalid("skip")
    stagesArray.append(assembler)

    val trainer = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("assemble")
      .setMaxIter(20)
    stagesArray.append(trainer)

    val pipeline = new Pipeline()
      .setStages(stagesArray.toArray)

    val model = pipeline.fit(training)

    val output = "hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang/risk"
    model.write.overwrite().save(output + "/piperisk_20210804")

    val traindf = model.transform(training)
    traindf.select("label", "prediction")
      .createOrReplaceTempView("traindf")
    getIndicators(spark, "traindf")

    val testpred = model.transform(test)
    val testdf = testpred.cache()
    testdf.select("label", "prediction")
      .createOrReplaceTempView("testdf")
    getIndicators(spark, "testdf")

    import spark.implicits._

    //调整阈值
    testdf.select("label", "prediction", "probability").rdd.map(p => {
      val label = p.getAs[Int](0)
      val prediction = if (p.getAs[DenseVector](2)(1) >= 0.7) 1.0 else 0.0
      (label.toDouble, prediction)
    }).toDF("label", "prediction")
      .createOrReplaceTempView("testDF_7")
    getIndicators(spark, "testDF_7")

    //调整阈值2
    testdf.select("label", "prediction", "probability").rdd.map(p => {
      val label = p.getAs[Int](0)
      val prediction = if (p.getAs[DenseVector](2)(1) >= 0.8) 1.0 else 0.0
      (label.toDouble, prediction)
    }).toDF("label", "prediction")
      .createOrReplaceTempView("testDF_8")
    getIndicators(spark, "testDF_8")


    //调整阈值3
    testdf.select("label", "prediction", "probability").rdd.map(p => {
      val label = p.getAs[Int](0)
      val prediction = if (p.getAs[DenseVector](2)(1) >= 0.9) 1.0 else 0.0
      (label.toDouble, prediction)
    }).toDF("label", "prediction")
      .createOrReplaceTempView("testDF_9")
    getIndicators(spark, "testDF_9")

    val evaluator = new BinaryClassificationEvaluator()
    evaluator.setMetricName("areaUnderROC")
    val testAuc= evaluator.evaluate(testdf)
    println(" test auc:" + testAuc)

    val gbtModel = model.stages(11).asInstanceOf[GBTClassificationModel]
    println("importance:"+ gbtModel.featureImportances)

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
                count(*) as predict_cnt
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
    val dataFrame = pos_data.union(neg_data.sample(false, ratio * 90))
    dataFrame
  }


  def tf_idf_stage(stagesArray:ListBuffer[PipelineStage], input: String, numFeatures: Int): Unit ={
    val tokenizer = new RegexTokenizer()
      .setInputCol(input)
      .setOutputCol(input + "_token")
      .setPattern(",")
    stagesArray.append(tokenizer)

    val hashingTF = new HashingTF()
      .setInputCol(input + "_token").setOutputCol(input + "_tf").setNumFeatures(numFeatures)
    stagesArray.append(hashingTF)

    val idf = new IDF().setInputCol(input + "_tf").setOutputCol(input + "_vec")
    stagesArray.append(idf)

  }
}
