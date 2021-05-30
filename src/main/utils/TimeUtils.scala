package utils

import java.text.SimpleDateFormat
import java.util.{Calendar, Date}

object TimeUtils {
    def getLostDays(updateTime: Date): Int = {
      val today = new Date()
      val diffDays = (today.getTime() - updateTime.getTime()) / (1000 * 60 * 60 * 24)
      (diffDays + 0.5).toInt
    }

    /**
      *
      * @param pattern   时间模板
      * @param deltaDays 日期差
      * @return
      */
    def getDateString(pattern: String = "yyyyMMdd", deltaDays: Int = 0): String = {
      val dateToday = Calendar.getInstance()
      dateToday.add(Calendar.DATE, deltaDays)
      val timeFormat: SimpleDateFormat = new SimpleDateFormat(pattern)
      timeFormat.format(dateToday.getTime)
    }

  /**
    * 当前时间增加天数活减少天数
    * @param dt
    * @param deltaDays
    * @return
    */
    def addDate(dt: String, deltaDays: Long = 0): String = {
      val fm = new SimpleDateFormat("yyyyMMdd")
      val date = fm.parse(dt)
      val addTime = date.getTime() + deltaDays * 86400 * 1000
      println(addTime)
      changFormat(fm.format(new Date(addTime)))
    }

  /**
    * yyyyMMdd -> yyyy-MM-dd
    * @param dt
    * @return
    */
    def changFormat(dt:String): String={
      val fm1 = new SimpleDateFormat("yyyyMMdd")
      val fm2 = new SimpleDateFormat("yyyy-MM-dd")
      fm2.format(new Date(fm1.parse(dt).getTime()))
    }

  def main(args: Array[String]): Unit = {
    println(addDate("20210323", 30))
    println(changFormat("20210508"))
    println(1528783 * 1.0/481873671)
  }
}
