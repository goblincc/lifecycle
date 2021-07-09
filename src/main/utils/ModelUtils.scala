package utils

import org.apache.spark.sql.{DataFrame, SparkSession}

object ModelUtils {
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
