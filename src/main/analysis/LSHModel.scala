package analysis

import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col

object LSHModel {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .config("spark.hadoop.validateOutputSpecs", value = false)
      .enableHiveSupport()
      .getOrCreate()
    spark.sparkContext.setLogLevel("error")

    val sqltxt =
      s"""
         |SELECT a.hdid,
         |		mate_id,
         |		if(is_low_cost_retain is null, 2, is_low_cost_retain) as is_low_cost_retain,
         |		if(is_profit_retain is null, 2, is_profit_retain) as is_profit_retain,
         |		if(sex is null, 'other', sex) as sex,
         |		if(age is null, 'other', age) as age,
         |		if(ip_city_level is null, 'other', ip_city_level) as ip_city_level,
         |		if(bd_consum is null, 'other', bd_consum) as bd_consum,
         |		if(bd_marriage is null, 'other', bd_marriage) as bd_marriage,
         |		if(bd_subconsum is null, 'other', bd_subconsum) as bd_subconsum,
         |		nvl(sys, 2) as sys,
         |		if(start_cnt_d1 is null, 0, start_cnt_d1) as start_cnt_d1,
         |		if(start_cnt_d3 is null, 0, start_cnt_d3) as start_cnt_d3,
         |		if(start_cnt_d7 is null, 0, start_cnt_d7) as start_cnt_d7,
         |		if(active_days_d1 is null, 0, active_days_d1) as active_days_d1,
         |		if(active_days_d3 is null, 0, active_days_d3) as active_days_d3,
         |		if(active_days_d7 is null, 0, active_days_d7) as active_days_d7,
         |		if(total_watch_dr_d1 is null, 0, total_watch_dr_d1) as total_watch_dr_d1,
         |		if(total_watch_dr_d3 is null, 0, total_watch_dr_d3) as total_watch_dr_d3,
         |		if(total_watch_dr_d7 is null, 0, total_watch_dr_d7) as total_watch_dr_d7,
         |		if(consume_cnt_d1 is null, 0, consume_cnt_d1) as consume_cnt_d1,
         |		if(consume_cnt_d3 is null, 0, consume_cnt_d3) as consume_cnt_d3,
         |		if(consume_cnt_d7 is null, 0, consume_cnt_d7) as consume_cnt_d7,
         |       if(d.hdid IS NULL, "", d.applist) AS applists,
         |       if(b.sjp IS NULL, 'other', b.sjp) AS sjp,
         |       if(c.hdid IS NULL, 1, live_cnt) AS live_cnt
         |FROM
         |  (SELECT *
         |   FROM persona.yylive_dws_profit_user_retain_feature
         |   WHERE dt >= '2021-05-16'
         |     AND dt <= '2021-06-16' and mate_id is not null) AS a
         |LEFT JOIN
         |  (SELECT sjp
         |   FROM persona.yylive_dws_user_sjp_rank
         |   WHERE rank <= 30) AS b ON lower(a.sjp) = lower(b.sjp)
         |LEFT JOIN
         |  (SELECT *
         |   FROM persona.yylive_dws_user_liveapp_cnt
         |   WHERE dt = '2021-06-23') AS c ON a.hdid = c.hdid
         |LEFT JOIN
         |  (SELECT hdid,
         |          applist
         |   FROM persona.yylive_dwd_applist_text
         |   WHERE dt="2021-06-16"
         |     AND TYPE=3) AS d ON a.hdid = d.hdid
       """.stripMargin

    val data = spark.sql(sqltxt)

    val num_feature = Array("start_cnt_d1","start_cnt_d3", "start_cnt_d7", "active_days_d1", "active_days_d3", "active_days_d3",
      "active_days_d7", "total_watch_dr_d1", "total_watch_dr_d3","total_watch_dr_d7", "consume_cnt_d1",
      "consume_cnt_d3", "consume_cnt_d7")


    val formula =
      s"""
         |is_low_cost_retain ~ sex + age + ip_city_level + bd_consum + bd_marriage + bd_subconsum + sjp
       """.stripMargin

    val rformula = new RFormula()
      .setFormula(formula)
      .setFeaturesCol("catVec")
      .setLabelCol("label")
      .setHandleInvalid("skip")

    val catVec = rformula.fit(data).transform(data)

    val tokenizer = new RegexTokenizer()
      .setInputCol("applists")
      .setOutputCol("words")
      .setPattern("\\|")
    val wordsData = tokenizer.transform(catVec)
//    wordsData.show(5, false)

    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(10)

    val featurizedData = hashingTF.transform(wordsData)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("appVec")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)

    val assembler = new VectorAssembler()
      .setInputCols(Array("catVec", "appVec") ++ num_feature)
      .setOutputCol("assemble")

    val assembleData = assembler.transform(rescaledData)

    val scaler = new MinMaxScaler()
      .setInputCol("assemble")
      .setOutputCol("features")

    val scalerModel = scaler.fit(assembleData)
    val output = scalerModel.transform(assembleData)

    import spark.implicits._

    val targetUser = output.where("is_low_cost_retain = 1 or is_profit_retain = 1")
                      .agg(Summarizer.mean($"features").alias("means"))

    val meanData: DataFrame = output.groupBy($"mate_id").agg(Summarizer.mean($"features").alias("means"))
    meanData.createOrReplaceTempView("meanData_db")
    spark.sql(
      s"""
         |insert overwrite table persona.yylive_quli_mean_feature partition(dt='2021-06-01')
         |	select mate_id, means from meanData_db
       """.stripMargin)

    val brp = new BucketedRandomProjectionLSH()
      .setBucketLength(0.2)
      .setNumHashTables(3)
      .setInputCol("means")
      .setOutputCol("bucketId")

    val model = brp.fit(meanData)

    // Feature Transformation
    println("The hashed dataset where hashed values are stored in the column 'hashes':")
    model.transform(meanData).show()

    val df = meanData.where("mate_id in('443639','440524','440508','440507','430999')")

    val candidate_sql =
      s"""
         |select mate_id from (
         |SELECT a.mate_id,
         |       row_number() over (partition BY 1 ORDER BY count(distinct hdid) desc) AS rank
         |FROM
         |  (SELECT mate_id,
         |          hdid
         |   FROM persona.yylive_dws_profit_user_retain_feature) AS a
         |LEFT JOIN
         |  (SELECT mate_id,
         |          count(DISTINCT hdid) num
         |   FROM persona.yylive_dws_profit_user_retain_feature
         |   WHERE dt >= '2021-05-16'
         |     AND dt <= '2021-06-16' and (is_low_cost_retain IS NOT NULL
         |          OR is_profit_retain IS NOT NULL)
         |   GROUP BY mate_id) AS b ON a.mate_id = b.mate_id
         |WHERE b.mate_id IS NULL and a.mate_id is not null GROUP BY a.mate_id
         |) as a where rank <= 50
       """.stripMargin
    val candidates = spark.sql(candidate_sql)
    val candidate = candidates.join(meanData,"mate_id").select("mate_id", "means")
//    candidate.show(5, false)

    val result = model.approxSimilarityJoin(candidate, df, 100000, "EuclideanDistance")
      .select(col("datasetA.mate_id").alias("idA"),
        col("datasetB.mate_id").alias("idB"),
        col("EuclideanDistance").alias("dist"))
    result.show(5, false)

    result.createOrReplaceTempView("result_db")
    spark.sql(
      s"""
         |insert overwrite table persona.yylive_quli_result_day partition(dt='2021-06-01')
         |	select idA,idB,dist from result_db
       """.stripMargin)

    val result2 = model.approxSimilarityJoin(candidate, targetUser, 100000, "EuclideanDistance")
      .select(col("datasetA.mate_id").alias("idA"),
        col("EuclideanDistance").alias("dist"))

    result2.createOrReplaceTempView("user_db")
    spark.sql(
      s"""
         |insert overwrite table persona.yylive_quli_result_user_target partition(dt='2021-06-01')
         |	select idA, dist from user_db
       """.stripMargin)


    spark.close()
  }

}
