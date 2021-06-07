package analysis

import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.sql.SparkSession

object corrAnalysis {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .config("spark.hadoop.validateOutputSpecs", value = false)
      .enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("warn")
    registUDF(spark)
    val data = spark.sql(
      s"""
         |select hdid, bd_sex, bd_consume, bd_marriage, featureVec(bd_sex,bd_consume,bd_marriage) as features,
         | if(sing_dr >= 300, 1, 0) as label_1,
         | if(game_dr >= 300, 1, 0) as label_2
         |  from persona.spring_activity_baidu_userprofile_analyse
         |where bd_sex is not null and bd_consume is not null and bd_marriage is not null
       """.stripMargin)

    val selector = new ChiSqSelector()
      .setNumTopFeatures(1)
      .setFeaturesCol("features")
      .setLabelCol("label_1")
      .setOutputCol("selectedFeatures")

    val result = selector.fit(data).transform(data)
    result.show()

    val chi = ChiSquareTest.test(data, "features", "label_1")
    chi.show(false)


    val chi_2 = ChiSquareTest.test(data, "features", "label_2")
    chi_2.show(false)

    spark.close()
  }

  def registUDF(spark: SparkSession): Unit = {
    spark.udf.register("featureVec", (s1: String, s2: String, s3: String) => {
      val f1 = if(s1 == "F") 0.0 else 1.0
      var f2 = 0.0
      if(s2 == "低"){
        f2 = 0.0
      }else if (s2 == "中"){
        f2 = 1.0
      }else{
        f2 = 2.0
      }

      var f3 = if(s3 == "已婚") 0.0 else 1.0

      Vectors.dense(f1, f2, f3)
    })

  }
}
