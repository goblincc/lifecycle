package model

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

object ALSModelTrain {

  val output = "hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang/ALS"

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .config("spark.hadoop.validateOutputSpecs", value = false)
      .enableHiveSupport()
      .getOrCreate()
    spark.sparkContext.setLogLevel("warn")
    import spark.implicits._
    val sc = spark.sparkContext
    val sqlTxt =
      s"""
         |SELECT hdid, docid, score FROM persona.yylive_lunxun_doc_hdid_score  WHERE dt="2021-05-28"
       """.stripMargin

    val data = spark.sql(sqlTxt)

    val click_rdd: RDD[(String, String, Double)] = data.rdd.map(p => {
      (p.getAs[String](0), p.getAs[String](1), p.getAs[Float](2).toDouble)
    })
    val user2Int: RDD[(String, Int)] = click_rdd.map(_._1).distinct().zipWithUniqueId().map(p => (p._1, p._2.toInt)).persist(StorageLevel.MEMORY_AND_DISK)
    val doc2Int: RDD[(String, Int)] = click_rdd.map(_._2).distinct().zipWithUniqueId().map(p => (p._1, p._2.toInt)).persist(StorageLevel.MEMORY_AND_DISK)

    user2Int.toDF("hdid", "index").createOrReplaceTempView("user")
    doc2Int.toDF("docid","index").createOrReplaceTempView("doc")

    spark.sql(
      s"""
         |insert overwrite table persona.yylive_dws_user_index partition(dt='2021-05-28')
         |	select * from user
       """.stripMargin)

    spark.sql(
      s"""
         |insert overwrite table persona.yylive_dws_doc_index partition(dt='2021-05-28')
         |	select * from doc
       """.stripMargin)

    val bc_user2Int = sc.broadcast(user2Int.collectAsMap())
    val bc_doc2Int = sc.broadcast(doc2Int.collectAsMap())

    val row: RDD[Row] = click_rdd.map(p => {
      val userId = bc_user2Int.value.getOrElse(p._1, 0)
      val docId = bc_doc2Int.value.getOrElse(p._2, 0)
      Row(userId, docId, p._3)
    })

    val structFields = Array(
      StructField("userId",IntegerType,true),
      StructField("itemId",IntegerType,true),
      StructField("rating",DoubleType,true)
    )

    val structType = DataTypes.createStructType(structFields)
    val df: DataFrame = spark.createDataFrame(row,structType)

    /**
      * numBlocks 是用于并行化计算的分块个数。
      * rank 是模型中隐语义因子的个数。就是平时的特征向量的长度。
      * maxIter：iterations 是迭代的次数。
      * lambda 是ALS的正则化参数。
      * implicitPrefs 决定了是用显性反馈ALS的版本还是用适用隐性反馈数据集的版本，如果是隐性反馈则需要将其参数设置为true。
      * alpha 是一个针对于隐性反馈 ALS 版本的参数，这个参数决定了偏好行为强度的基准。
      * itemCol:deal的字段名字，需要跟表中的字段名字是一样的。
      * nonnegative:是否使用非负约束，默认不使用 false。
      * predictionCol:预测列的名字
      * ratingCol：评论字段的列名字，要跟表中的数据字段一致。
      * userCol：用户字段的名字，同样要保持一致。
      */
    val als = new ALS()
      .setImplicitPrefs(true)
      .setRank(10)
      .setAlpha(0.1)
      .setMaxIter(20)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("itemId")
      .setRatingCol("rating")
    val model = als.fit(df)

    val predictions = model.transform(df)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")

    model.itemFactors.show(5, false)
    model.save(output + "/AlsModel")

  }


}
