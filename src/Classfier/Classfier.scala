import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, Word2Vec}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}

object Classfier {
  final val VECTOR_SIZE = 100

  def main(args: Array[String]) {

//            if (args.length < 1) {
//                println("Usage:SMSClassifier SMSTextFile")
//                sys.exit(1)
//            }
//    val filepath = "file:///home/rr/文档/SMSSpamCollection"
    val filepath = args(0)
    val conf = new SparkConf().setAppName("Message Classification")
    val sc = new SparkContext(conf)
    val parsedRDD = sc.textFile(filepath).map(_.split("\t")).map(str => {
      (str(0), str(1).split(" "))
    })

    val sqlCtx = new SQLContext(sc)
    val msgDF = sqlCtx.createDataFrame(parsedRDD).toDF("label", "message")
    //    将文字型标签转换成数值型标签
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(msgDF)
    //     将消息内容变成数值型向量
    val word2Vec = new Word2Vec()
      .setInputCol("message")
      .setOutputCol("features")
      .setVectorSize(VECTOR_SIZE)
      .setMinCount(1)
    //     构建多层前馈神经网络模型
    val layers = Array[Int](VECTOR_SIZE, 10,2)
    val ml = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(512)
      .setSeed(1234L)
      .setMaxIter(128)
      .setFeaturesCol("features")
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
    //     将数值型标签变为文字型标签
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)
    //     构造训练集和机器学习工作流
    val Array(trainingData, testData) = msgDF.randomSplit(Array(0.8, 0.2), 2L)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, word2Vec, ml, labelConverter))

    //    对模型进行训练和测试
    val model = pipeline.fit(trainingData)
    val predictionResultDF = model.transform(testData)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val evaluator1 = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("weightedPrecision")

    val evaluator2 = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    val predictionAccuracy = evaluator.evaluate(predictionResultDF)
    val precision = evaluator1.evaluate(predictionResultDF)
    val f = evaluator2.evaluate(predictionResultDF)

    predictionResultDF.select("message", "label", "predictedLabel").show(1100)
    predictionResultDF.printSchema


    println("Testing Accuracy is %2.4f".format(predictionAccuracy * 100) + "%")
    println("Testing Precision is %2.4f".format(precision * 100) + "%")
    println("Testing F is %2.4f".format(f * 100) + "%")
    sc.stop
  }
}
