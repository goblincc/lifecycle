package analysis

import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
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
      .master("local[*]")
      .enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("error")

    val customSchema = StructType(Array(
      StructField("label", IntegerType, true),
      StructField("dtcnt", IntegerType, true),
      StructField("all_dt", IntegerType, true),
      StructField("amount", IntegerType, true),
      StructField("amount_avg", DoubleType, true),
      StructField("chid_cnt", IntegerType, true),
      StructField("paymethod_cnt", IntegerType, true),
      StructField("userip_cnt", IntegerType, true),
      StructField("max_cnt_dt", IntegerType, true),
      StructField("avg_cnt_dt", DoubleType, true))
    )

    val data = spark
      .read
      .format("csv")
      .option("header","true")
      .option("multiLine", true)
//      .schema(customSchema)
      .load("./data/risk.csv")

    data.createOrReplaceTempView("df_table")
    val df = spark.sql(
      s"""
         |select
         |cast(label as int) as label,
         |cast(dtcnt as int) as dtcnt,
         |cast(all_dt as int) as all_dt,
         |cast(amount as int) as amount,
         |cast(amount_avg as float) as amount_avg,
         |cast(chid_cnt as int) as chid_cnt,
         |cast(paymethod_cnt as int) as paymethod_cnt,
         |cast(userip_cnt as int) as userip_cnt,
         |cast(max_cnt_dt as int) as max_cnt_dt,
         |cast(avg_cnt_dt as float) as avg_cnt_dt
         |from df_table
       """.stripMargin)

    val splits = df.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = sampleData(splits(0))
    println("train_pos:"+ training.where("label = 1.0").count())
    println("train_neg:"+ training.where("label = 0.0").count())
    val test = splits(1)
    println("test_pos:"+ test.where("label = 1.0").count())
    println("test_neg:"+ test.where("label = 0.0").count())

    val num_feature = Array("dtcnt", "all_dt", "amount", "amount_avg", "chid_cnt", "paymethod_cnt", "userip_cnt", "max_cnt_dt", "avg_cnt_dt")

    val stagesArray = new ListBuffer[PipelineStage]()

    val assembler = new VectorAssembler()
      .setInputCols(num_feature)
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

    val traindf = model.transform(training)

    traindf.show(10,false)
    traindf.select("label", "prediction")
      .createOrReplaceTempView("traindf")
    getIndicators(spark, "traindf")

    val testdf = model.transform(test)
    testdf.show(10,false)
    testdf.select("label", "prediction")
      .createOrReplaceTempView("testdf")
    getIndicators(spark, "testdf")

    val dataframe = testdf.select("label", "prediction", "probability").rdd.map(p => {
      (p.getAs[Int](0), p.getAs[Double](1), p.getAs[DenseVector](2)(1))
    }).toDF("label", "prediction", "probability")
    dataframe.show(5, false)
    aucCal(dataframe)

    val evaluator = new BinaryClassificationEvaluator()
    evaluator.setMetricName("areaUnderROC")
    val testAuc= evaluator.evaluate(testdf)
    println(" test auc:" + testAuc)
    val gbtModel = model.stages(1).asInstanceOf[GBTClassificationModel]
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
    val dataFrame = pos_data.union(neg_data.sample(false, ratio * 50))
    dataFrame
  }

  def aucCal(df: DataFrame):Double= {
    val sort: RDD[((Int, Double, Double), Long)] = df.rdd.map(p => {
      (p.getAs[Int](0), p.getAs[Double](1), p.getAs[Double](2))
    }).sortBy(_._3, ascending = true).zipWithIndex()
    //计算正样本的ranker之和
    val posSum = sort.filter(_._1._1 == 1).map(_._2).sum()
    //计算正样本数量M和负样本数量N
    val M = sort.filter(_._1._1 == 1).count
    val N = sort.filter(_._1._1 == 0).count
    //计算公式
    val auc = (posSum - ((M - 1.0) * M) / 2) / (M * N)
    println("aucCal:" + auc)
    auc
  }
}
