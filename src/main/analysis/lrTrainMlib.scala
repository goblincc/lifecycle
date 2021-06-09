package analysis
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer

object lrTrainMlib {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder()
      .config("spark.hadoop.validateOutputSpecs", value = false)
      .enableHiveSupport()
      .getOrCreate()
    sc.setLogLevel("warn")

    val data = spark.sql(
      s"""
         select
         |if(sing_dr > 300, 1, 0) as label,
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
         |and bd_dance is not Null
         |and bd_talk is not NULL
         |and bd_outdoor is not NULL
         |and bd_game is not NULL
         |and bd_sport is not NULL
         |and sing_dr > 0
       """.stripMargin)

    data.rdd.map(p => {
      val res = new ArrayBuffer[(Int, Double)]()
      val bd_sex = p.getAs[String]("bd_sex")
      val bd_age = p.getAs[String]("bd_age")
      val bd_consume = p.getAs[String]("bd_consume")
      val bd_marriage = p.getAs[String]("bd_marriage")
      val bd_high_value = p.getAs[String]("bd_high_value")
      val bd_low_value = p.getAs[String]("bd_low_value")
      val bd_sing = p.getAs[String]("bd_sing")
      val bd_dance = p.getAs[String]("bd_dance")
      val bd_talk = p.getAs[String]("bd_talk")
      val bd_outdoor = p.getAs[String]("bd_outdoor")
      val bd_game = p.getAs[String]("bd_game")
      val bd_sport = p.getAs[String]("bd_sport")

      if (bd_sex == "M") {
        res.append((1, 1.0))
      } else {
        res.append((2, 1.0))
      }

      if (bd_age == "35-44") {
        res.append((3, 1.0))
      } else if (bd_age == "35-44") {
        res.append((4, 1.0))
      } else if (bd_age == "18-24") {
        res.append((5, 1.0))
      } else if (bd_age == "45-54") {
        res.append((6, 1.0))
      } else if (bd_age == "65以上") {
        res.append((7, 1.0))
      } else if (bd_age == "18以下") {
        res.append((8, 1.0))
      } else if (bd_age == "25-34") {
        res.append((7, 1.0))
      } else if (bd_age == "55-64") {
        res.append((10, 1.0))
      }

      if (bd_consume == "高") {
        res.append((11, 1.0))
      } else if (bd_consume == "中") {
        res.append((12, 1.0))
      } else if (bd_consume == "低") {
        res.append((13, 1.0))
      }

      if (bd_marriage == "已婚") {
        res.append((14, 1.0))
      } else if (bd_marriage == "未婚") {
        res.append((15, 1.0))
      }

      if (bd_high_value == "Y") {
        res.append((16, 1.0))
      }

      if (bd_low_value == "Y") {
        res.append((17, 1.0))
      }

      if (bd_sing == "Y") {
        res.append((18, 1.0))
      }

      if (bd_dance == "Y") {
        res.append((19, 1.0))
      }

      if (bd_talk == "Y") {
        res.append((20, 1.0))
      }

      if (bd_outdoor == "Y") {
        res.append((21, 1.0))
      }

      if (bd_game == "Y") {
        res.append((22, 1.0))
      }

      if (bd_sport == "Y") {
        res.append((23, 1.0))
      }
      val label = p.getAs[Int]("label")

//    LabeledPoint(label.toDouble, Vectors.dense(res.sortWith(_._1 < _._1).map(p => p._2).toArray))
      label + " " + res.sortWith(_._1 < _._1).map(p => p._1 + ":" + p._2).mkString(" ")
    }).repartition(1).saveAsTextFile("hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang/lbsvm/lbsvm_1")

    val trainData = MLUtils.loadLibSVMFile(sc, "hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang/lbsvm/lbsvm_1",23)
    val lrModel = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)

    val model = lrModel.run(trainData)

    val predictionAndLabels = trainData.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    val metrics = new MulticlassMetrics(predictionAndLabels)
    val accuracy = metrics.accuracy
    println(s"Accuracy = $accuracy")

    val metrics2 = new BinaryClassificationMetrics(predictionAndLabels)
    val auROC = metrics2.areaUnderROC()
    val value = metrics2.precisionByThreshold()
    println(s"auRoc = $auROC")
    println(s"precision = $value")

    val weights: linalg.Vector = model.weights

    println("weights:" + weights)
  }
}
