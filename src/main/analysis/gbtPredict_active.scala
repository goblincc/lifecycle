package analysis

import model.DocCTRPredict.{modelPath, saveData2hive}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DataTypes, DoubleType, StringType, StructField}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import utils.TimeUtils

object gbtPredict_active {
  val modelPath = "hdfs://yycluster02/hive_warehouse/persona_client.db/chenchang/pipe"
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("warn")

    val sqlTxt =
      s"""
         |select * ,
         |'1' as is_exposure_v2
         |from persona.yylive_baoxiang_feature_day where dt = '2021-07-07' and last_active_dt >= date_add(dt, -3)
       """.stripMargin
    print("sqlTxt:", sqlTxt)

    val data = spark.sql(sqlTxt)


    val sqlTxt2 =
      s"""
         |select * ,
         |'0' as is_exposure_v2
         |from persona.yylive_baoxiang_feature_day where dt = '2021-07-07' and last_active_dt >= date_add(dt, -3)
       """.stripMargin
    print("sqlTxt2:", sqlTxt2)

    val data2 = spark.sql(sqlTxt2)

    val model = PipelineModel.read.load( modelPath + "pipelineActive_2021-06-15" )
    val predict = model.transform(data)
    val predict2 = model.transform(data2)
    saveData2hive(spark, "2021-06-01", predict,"1")
    saveData2hive(spark, "2021-06-01", predict2, "0")
  }

  def saveData2hive(spark:SparkSession, dt: String, dataFrame: DataFrame,dtype: String): Unit ={
    val structFields = Array(
      StructField("hdid",StringType,true),
      StructField("docid",StringType,true),
      StructField("prediction",DoubleType,true),
      StructField("probability",DoubleType,true)
    )
    val structType = DataTypes.createStructType(structFields)
    val row: RDD[Row] = dataFrame.select("hdid", "doc_id","prediction", "probability").rdd.map(p => {
      val hdid = p.getString(0)
      val docid = p.getString(1)
      val prediction = p.getDouble(2)
      val probability = p.getAs[DenseVector](3)(1)
      Row(hdid, docid, prediction, probability)
    })

    spark.createDataFrame(row,structType).createOrReplaceTempView("tb_save")

    spark.sql(
      s"""
         |insert overwrite table persona.yylive_baoxiang_predict_info partition(dt='${dt}',dtype=0)
         |	select * from tb_save
       """.stripMargin)

  }

}
