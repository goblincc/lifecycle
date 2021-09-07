package graphx

import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import utils.TimeUtils

import scala.io.Source

object PregelDemo {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .config("spark.hadoop.validateOutputSpecs", value = false)
      .master("local[1]")
      .enableHiveSupport()
      .getOrCreate()
    spark.sparkContext.setLogLevel("warn")
    val sc = spark.sparkContext
    val users: RDD[(VertexId, String)] =
      sc.parallelize(Array(
        (3L, "rxin"),
        (7L, "jgonzal"),
        (5L, "franklin"),
        (2L, "istoica"),
        (4L, "peter"),
        (10L, "lucy"),
        (16L, "tom")
      ))

    val relationships: RDD[Edge[String]] =
      sc.parallelize(Array(
        Edge(3L, 7L, "collab"),
        Edge(5L, 3L, "advisor"),
        Edge(2L, 5L, "colleague"),
        Edge(5L, 7L, "pi"),
        Edge(4L, 10L, "student"),
        Edge(5L, 0L, "colleague"),
        Edge(0L, 2L, "colleague"),
        Edge(7L, 16L, "colleague")
      ))

    val graph = Graph(users, relationships)

    //定义n度关系
    val n = 3

    //每个节点开始的时候只存储了（该节点编号，n）这一个键值对
    val newG = graph.mapVertices((vid, _) => Map[VertexId, Int](vid -> n))
      .pregel(Map[VertexId, Int](), n, EdgeDirection.Out)(vprog, sendMsg, mergeMsg)

    newG.vertices.mapValues(_.filter(_._2 == 0).keySet)
      .filter(_._2 != Set())
      .map(p=>p._1 + "节点的"+ n + "度关系节点-->" + p._2.mkString(","))
      .foreach(println(_))
  }

  /**
    * 更新节点数据，vdata为本身数据，message为消息数据
    */
  def vprog(vid: VertexId, vdata: Map[VertexId, Int], message: Map[VertexId, Int])
  : Map[VertexId, Int] = {
    mergeMsg(vdata, message)
  }

  /**
    * 节点更新数据发送消息
    */
  def sendMsg(e: EdgeTriplet[Map[VertexId, Int], _]) = {
    val srcMap = (e.dstAttr.keySet -- e.srcAttr.keySet).map { k => k -> (e.dstAttr(k) - 1) }.toMap
    val dstMap = (e.srcAttr.keySet -- e.dstAttr.keySet).map { k => k -> (e.srcAttr(k) - 1) }.toMap
    if (srcMap.isEmpty && dstMap.isEmpty)
      Iterator.empty
    else
      Iterator((e.dstId, dstMap), (e.srcId, srcMap))
  }

  /**
    * 对于交集的点的处理，取msg1和msg2中最小的值
    */
  def mergeMsg(msg1: Map[VertexId, Int], msg2: Map[VertexId, Int]): Map[VertexId, Int] =
    (msg1.keySet ++ msg2.keySet).map {
      k => k -> math.min(msg1.getOrElse(k, Int.MaxValue), msg2.getOrElse(k, Int.MaxValue))
    }.toMap
}
