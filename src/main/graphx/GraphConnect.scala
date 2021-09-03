package graphx

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx.{Edge, Graph, VertexId, VertexRDD}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

object GraphConnect {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .config("spark.hadoop.validateOutputSpecs", value = false)
      .enableHiveSupport()
      .getOrCreate()

    spark.sparkContext.setLogLevel("error")
    import spark.implicits._
    val sc = spark.sparkContext
    val sqlTxt =
      s"""
         |select uid as vertices, "uid" as types from persona.yylive_uid_mobile_info where dt = '2021-09-01'
         |union
         |select mobile as vertices, "mobile" as types from persona.yylive_uid_mobile_info where dt = '2021-09-01'
         |union
         |select uid as vertices, "uid" as types from persona.yylive_uid_idnum_info where dt = '2021-09-01'
         |union
         |select idnum as vertices, "idnum" as types from persona.yylive_uid_idnum_info where dt = '2021-09-01'
       """.stripMargin

    val verticesDataFrame = spark.sql(sqlTxt).persist(StorageLevel.MEMORY_AND_DISK)
    verticesDataFrame.createOrReplaceTempView("table_vertices")

    val verticesRdd: RDD[(String, Long)] = verticesDataFrame.rdd.map(p => {
      p.getAs[String]("vertices")
    }).zipWithUniqueId()

    verticesRdd.toDF("vertices", "id").persist(StorageLevel.MEMORY_AND_DISK).createOrReplaceTempView("table_index")

    val vertices: RDD[(VertexId, String)] = verticesDataFrame.rdd.map(p => {
      (p.getString(0), p.getString(1))
    }).join(verticesRdd, 300).map(p => {
      (p._2._2, p._2._1)
    })

    val sqlTxt2 =
      s"""
         |select mobile, uid, "mobile_conn" as conn from persona.yylive_uid_mobile_info where dt = '2021-09-01'
         |union
         |select idnum, uid, "idnum_conn" as conn from persona.yylive_uid_idnum_info where dt = '2021-09-01'
       """.stripMargin

    spark.sql(sqlTxt2).toDF("src", "dst", "conn").createOrReplaceTempView("table_edge")

    val relationships: RDD[Edge[String]] = spark.sql(
      s"""
         |SELECT srcid,
         |       id AS dstid,
         |       conn
         |FROM
         |  (SELECT id AS srcid,
         |          dst,
         |          conn
         |   FROM table_edge AS a
         |   INNER JOIN table_index AS b ON a.src = b.vertices) AS a
         |INNER JOIN table_index AS b ON a.dst = b.vertices
       """.stripMargin).rdd.map(p=>{
      Edge(p.getLong(0), p.getLong(1), p.getString(2))
    })

    val graph = Graph(vertices, relationships)
    val connectVertices: VertexRDD[VertexId] = graph.connectedComponents().vertices
    connectVertices.toDF("verticesid","categoryid").createOrReplaceTempView("table_connect")

    val result = spark.sql(
      s"""
         |insert overwrite table persona.yylive_uid_risk_groups partition(dt='2021-09-01')
         |SELECT b.vertices,
         |       types,
         |       categoryid
         |FROM table_connect AS a
         |INNER JOIN table_index AS b ON a.verticesid = b.id
         |INNER JOIN table_vertices AS c ON b.vertices = c.vertices
       """.stripMargin)

  }

}
