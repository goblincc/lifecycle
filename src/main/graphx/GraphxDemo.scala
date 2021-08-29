package graphx
import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.graphx.util.GraphGenerators
//import org.apache.spark.graphx.{PartitionID, VertexId}
//import org.apache.spark.graphx._
import org.apache.spark.sql.SparkSession
// To make some of the examples work we will also need RDD
import org.apache.spark.rdd.RDD

object GraphxDemo {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local[1]").appName("Simple Application").getOrCreate()
    val sc = spark.sparkContext
    val graph: Graph[Double, Int] =
      GraphGenerators.logNormalGraph(sc, numVertices = 100).mapVertices( (id, _) => id.toDouble )
    // Compute the number of older followers and their total age
    graph.triplets.filter(p=>p.srcAttr > p.dstAttr).collect.foreach(println(_))
    val olderFollowers: VertexRDD[(Int, Double)] = graph.aggregateMessages[(Int, Double)](
      triplet => { // Map Function
        if (triplet.srcAttr > triplet.dstAttr) {
          // Send message to destination vertex containing counter and age
          triplet.sendToDst((1, triplet.srcAttr))
        }
      },
      // Add counter and age
      (a, b) => (a._1 + b._1, a._2 + b._2) // Reduce Function
    )
    // Divide total age by number of older followers to get average age of older followers
    val avgAgeOfOlderFollowers: VertexRDD[Double] =
      olderFollowers.mapValues( (id, value) =>
        value match { case (count, totalAge) => totalAge / count } )
    // Display the results
    avgAgeOfOlderFollowers.collect.foreach(println(_))
  }
}
