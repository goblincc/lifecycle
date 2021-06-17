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
         |select
         |if(talk_dr > 137, 1, 0) as labels,
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
    spark.udf.register("featureVec", (s1: String, s2: String, s3: String,s4: String,s5: String,s6: String,s7:
    String,s8: String,s9: String,s10: String, s11: String, s12: String) => {
      val sex = if(s1 == "F") 0.0 else 1.0

      var age = 0.0
      if (s2 == "35-44") {
        age = 1.0
      } else if (s2 == "35-44") {
        age = 2.0
      } else if (s2 == "18-24") {
        age = 3.0
      } else if (s2 == "45-54") {
        age = 4.0
      } else if (s2 == "65以上") {
        age = 5.0
      } else if (s2 == "18以下") {
        age = 6.0
      } else if (s2 == "25-34") {
        age = 7.0
      } else if (s2 == "55-64") {
        age = 8.0
      }

      var consume = 0.0
      if(s3 == "低"){
        consume = 0.0
      }else if (s3 == "中"){
        consume = 1.0
      }else{
        consume = 2.0
      }
      val marrage = if(s4 == "已婚") 0.0 else 1.0
      val bd_high_value = if(s5=="Y") 0.0 else 1.0
      val bd_low_value = if(s6=="Y") 0.0 else 1.0
      val bd_sing = if(s7=="Y") 0.0 else 1.0
      val bd_dance = if(s8=="Y") 0.0 else 1.0
      val bd_talk = if(s9=="Y") 0.0 else 1.0
      val bd_outdoor = if(s10=="Y") 0.0 else 1.0
      val bd_game = if(s11=="Y") 0.0 else 1.0
      val bd_sport = if(s12=="Y") 0.0 else 1.0
      Vectors.dense(sex, age, consume, marrage, bd_high_value, bd_low_value, bd_sing, bd_dance,bd_talk, bd_outdoor, bd_game,  bd_sport )
    })

  }
}
