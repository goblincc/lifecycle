package graphx

import org.apache.spark.graphx.{Edge, Graph, VertexId, VertexRDD}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object GraphxDemo2 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local[1]").appName("Simple Application").getOrCreate()
    val sc = spark.sparkContext
    val users: RDD[(VertexId, String)] =
      sc.parallelize(Array((3L, "rxin"), (7L, "jgonzal"),
        (5L, "franklin"), (2L, "istoica"),
        (4L, "peter"), (10L, "lucy")))
    // Create an RDD for edges
    val relationships: RDD[Edge[String]] =
      sc.parallelize(Array(Edge(3L, 7L, "collab"),    Edge(5L, 3L, "advisor"),
        Edge(2L, 5L, "colleague"), Edge(5L, 7L, "pi"),
        Edge(4L, 10L, "student"),   Edge(5L, 0L, "colleague"), Edge(0L, 2L, "colleague")))
    // Define a default user in case there are relationship with missing user
//    val defaultUser = ("John Doe", "Missing")

    val graph = Graph(users, relationships)
    val vertices: VertexRDD[VertexId] = graph.connectedComponents().vertices
    /*
      (4,4)
      (0,0)
      (3,0)
      (7,0)
      (10,4)
      (5,0)
      (2,0)
     */
//    是一个tuple类型，key分别为所有的顶点id，value为key所在的连通体id(连通体中顶点id最小值)
    /*vertices.foreach(println(_))
    users.join(vertices).map{
      case(id,(username,value))=>(value,username)
    }.groupByKey().map(t=>{
      t._1+"->"+t._2.mkString(",")
    }).foreach(println(_))
    import spark.implicits._
    vertices.toDF("id1", "id2").show(5,false)
    users.join(vertices).map(p=>{
      (p._2._2, p._2._1)
    }).foreach(println(_))*/

    val triCounts = graph.triangleCount().vertices
    triCounts.foreach(println(_))
  }

}
