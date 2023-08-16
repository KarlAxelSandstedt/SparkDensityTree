import scala.language.postfixOps

import org.apache.spark.mllib.linalg.{ Vector => MLVector, _ }
import scala.math.{abs, pow}

import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.simple.RandomSource;

import org.apache.spark.rdd.{PartitionPruningRDD, RDD}
import org.apache.spark.Partitioner
import org.apache.spark.sql.{ Dataset, SparkSession }
import org.apache.spark.{ SparkContext, SparkConf }
import org.apache.spark.rdd.RDD
import org.apache.log4j.{ Logger, Level }
import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD

import org.scalatest.{ path => testPath, _ }
import org.scalactic.TolerantNumerics

import co.wiklund.disthist._
import co.wiklund.disthist.MDEFunctions._
import co.wiklund.disthist.Types._
import co.wiklund.disthist.NodeLabelFunctions._
import co.wiklund.disthist.TruncationFunctions._
import co.wiklund.disthist.LeafMapFunctions._
import co.wiklund.disthist.RectangleFunctions._
import co.wiklund.disthist.SpatialTreeFunctions._
import co.wiklund.disthist.SplitEstimatorFunctions._
import co.wiklund.disthist.HistogramFunctions._
import co.wiklund.disthist.SubtreePartitionerFunctions._
import co.wiklund.disthist.MergeEstimatorFunctions._
import co.wiklund.disthist.BinarySearchFunctions._
import co.wiklund.disthist.UnfoldTreeFunctions._



class DensityTests extends FlatSpec with Matchers with BeforeAndAfterAll {
  // "it" should "compile" in {
  //   assert(1 === 1)
  // }
  private var sc : SparkContext = null
  private var df : RDD[MLVector] = null
  private var dfLocal : Vector[MLVector] = null
  private var bb : Rectangle = null
  private var h : Histogram = null
  private var h2 : Histogram = null
  private var tree : SpatialTree = null

  private val dfnum = 5000
  private val dfdim = 3
    // val df = normalVectorRDD(sc, n, 2)

  def lims(totalVolume : Double, totalCount : Count)(depth : Int, volume : Volume, count : Count) =
    count > dfnum/2 || (1 - count/totalCount)*volume > 0.001*totalVolume

  def limsC(totalVolume : Double, totalCount : Count)(depth : Int, volume : Volume, count : Count) =
    count > 10

  override protected def beforeAll() : Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val spark = SparkSession.builder.master("local").getOrCreate
    spark.conf.set("spark.default.parallelism", "6")
    sc = spark.sparkContext 
    df = normalVectorRDD(sc, dfnum, dfdim, 6, 7387389).cache()
    dfLocal = df.collect().toVector
    bb = boundingBox(df)
    tree = uniformTreeRootedAt(bb)
    h = histogram(df, lims)
    h2 = histogramStartingWith(h, df, limsC)
    assert(h2.counts.truncation != h.counts.truncation)
  }

  override protected def afterAll() : Unit = {
    df.unpersist()
    val spark = getSpark
    spark.stop
    sc = null
    df = null
    bb = null
  }

  private def getSpark: SparkSession = SparkSession.getActiveSession.get

  implicit val ordering : Ordering[NodeLabel] = leftRightOrd

  var density : DensityHistogram = null
  var finestResDepth : Depth = 0

  "quickSlice" should "give the same results as slice" in {
    val spark = getSpark
    import spark.implicits._
 
    println("Starting Regression test for quickSlice")
    val dimensions = 3
    val sizeExp = 5
    val numPartitions = 8
    spark.conf.set("spark.default.parallelism", numPartitions.toString)
    
    val trainSize = math.pow(10, sizeExp).toLong
    val finestResSideLength = 1e-5 
  
    val trainingRDD = normalVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions, 1230568)
    val validationRDD =  normalVectorRDD(spark.sparkContext, trainSize/2, dimensions, numPartitions, 5465694)
    val testRDD =  normalVectorRDD(spark.sparkContext, 1000, dimensions, numPartitions, 6949239)

  
    /* Get boxhull of training data and test data */
    var rectTrain = RectangleFunctions.boundingBox(trainingRDD)
    var rectValidation = RectangleFunctions.boundingBox(validationRDD)
    val rootBox = RectangleFunctions.hull(rectTrain, rectValidation)
  
    val tree = widestSideTreeRootedAt(rootBox)
    finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth
    val stepSize = math.ceil(finestResDepth / 8.0).toInt
    val kInMDE = 10
  
    /**
     * Label each datapoint by the label of the box at depth (finestResDepth) which it resides in, and count number of datapoints
     * found in occupied boxes.
     */
    var countedTrain = quickToLabeled(tree, finestResDepth, trainingRDD)
    var countedValidation = quickToLabeled(tree, finestResDepth, validationRDD)
        
    val sampleSizeHint = 100
    val partitioner = new SubtreePartitioner(numPartitions, countedTrain, sampleSizeHint)
    val depthLimit = partitioner.maxSubtreeDepth
    val countLimit = 100 
    val subtreeRDD = countedTrain.repartitionAndSortWithinPartitions(partitioner)
    val mergedHistogram = mergeLeavesHistogram(tree, subtreeRDD, countLimit, depthLimit)
  
    density = toDensityHistogram(getMDE(
      mergedHistogram,
      countedValidation, 
      trainSize/2,
      kInMDE, 
      false 
    )).normalize

    val testData = testRDD.collect  
    
    for (i <- 0 until testData.length) {

      val slicePoint : Vector[Double] = testData(i).toArray.toVector.drop(1) 
      val sliceAxes : Vector[Axis] = Vector(1,2) 

      val tree : WidestSplitTree = WidestSplitTree(density.tree.rootCell) 
      val splitOrder = tree.splitOrderToDepth(finestResDepth)
      var conditional = quickSlice(density, sliceAxes, slicePoint, splitOrder)

      if (conditional != null) {
        conditional = conditional.normalize

        val conditionalOld = slice(density, sliceAxes, slicePoint).normalize
        val leaves1 = conditional.densityMap.truncation.leaves
        val leaves2 = conditionalOld.densityMap.truncation.leaves
        val vals1 = conditional.densityMap.vals
        val vals2 = conditionalOld.densityMap.vals
  
        assert(leaves1.length == leaves2.length)
        for (j <- 0 until leaves1.length) {
          assert(leaves1(j) == leaves2(j))
          assert(vals1(j)._1 == vals2(j)._1)
          assert(vals1(j)._2 == vals2(j)._2)
        }
      }
    }
    println("Finished Regression test for quickSlice")
  }

  it should "return null on slicePoints outside the histogram's root box" in {

    val sliceAxes : Vector[Axis] = Vector(1,2) 
    val tree : WidestSplitTree = WidestSplitTree(density.tree.rootCell) 
    val splitOrder = tree.splitOrderToDepth(finestResDepth)

    val mid = Vector(
        (tree.rootCell.high(1) - tree.rootCell.low(1)) / 2.0,
        (tree.rootCell.high(2) - tree.rootCell.low(2)) / 2.0
      )
    val slicePoint1 = Vector(mid(0) + 100.0, mid(1))
    val slicePoint2 = Vector(mid(0) - 100.0, mid(1))
    val slicePoint3 = Vector(mid(0), mid(1) + 100.0)
    val slicePoint4 = Vector(mid(0), mid(1) - 100.0)

    assert(quickSlice(density, sliceAxes, slicePoint1, splitOrder) == null)
    assert(quickSlice(density, sliceAxes, slicePoint2, splitOrder) == null)
    assert(quickSlice(density, sliceAxes, slicePoint3, splitOrder) == null)
    assert(quickSlice(density, sliceAxes, slicePoint4, splitOrder) == null)
  }

  it should "return null on being given a slice point which only slice null-sets" in {
    val tree : WidestSplitTree = WidestSplitTree(Rectangle(Vector(0.0, 0.0, 0.0), Vector(2.0, 2.0, 2.0))) 
    val truncation = Truncation(Vector(NodeLabel(9), NodeLabel(15)))
    val values = Vector((0.5, 1.0), (0.5, 1.0))
    val boxDensity = DensityHistogram(tree, LeafMap(truncation, values))
    val splitOrder = tree.splitOrderToDepth(3)

    val sliceAxes1 = Vector(0,1)
    val slicePoint1 = Vector(0.5, 1.5)
    assert(quickSlice(boxDensity, sliceAxes1, slicePoint1, splitOrder) == null)

    val sliceAxes2 = Vector(0,1)
    val slicePoint2 = Vector(1.5, 0.5)
    assert(quickSlice(boxDensity, sliceAxes2, slicePoint2, splitOrder) == null)

    val sliceAxes3 = Vector(0,2)
    val slicePoint3 = Vector(0.5, 0.5)
    assert(quickSlice(boxDensity, sliceAxes3, slicePoint3, splitOrder) == null)

    val sliceAxes4 = Vector(2)
    val slicePoint4 = Vector(0.5)
    assert(quickSlice(boxDensity, sliceAxes4, slicePoint4, splitOrder) == null)
    
    val sliceAxes5 = Vector(1,2)
    val slicePoint5 = Vector(1.5, 0.5)
    assert(quickSlice(boxDensity, sliceAxes5, slicePoint5, splitOrder) == null)
  }

  it should "generate a conditional with correct values" in {
    val tree : WidestSplitTree = WidestSplitTree(Rectangle(Vector(0.0, 0.0, 0.0), Vector(2.0, 2.0, 2.0))) 
    val truncation = Truncation(Vector(NodeLabel(9), NodeLabel(15)))
    val values = Vector((0.5, 1.0), (0.5, 1.0))
    val boxDensity = DensityHistogram(tree, LeafMap(truncation, values))
    val splitOrder = tree.splitOrderToDepth(3)

    val sliceAxes1 = Vector(0,1)
    val slicePoint1 = Vector(0.5, 0.5)
    val density1 = quickSlice(boxDensity, sliceAxes1, slicePoint1, splitOrder)
    val leaves1 = density1.densityMap.truncation.leaves
    val values1 = density1.densityMap.vals
    assert(leaves1.length == 1)
    assert(leaves1(0) == NodeLabel(3))
    assert(values1(0) == (0.5, 1))

    val sliceAxes2 = Vector(0,2)
    val slicePoint2 = Vector(1.5, 1.5)
    val density2 = quickSlice(boxDensity, sliceAxes2, slicePoint2, splitOrder)
    val leaves2 = density2.densityMap.truncation.leaves
    val values2 = density2.densityMap.vals
    assert(leaves2.length == 1)
    assert(leaves2(0) == NodeLabel(3))
    assert(values2(0) == (0.5, 1))

    val sliceAxes3 = Vector(1,2)
    val slicePoint3 = Vector(0.5, 1.5)
    val density3 = quickSlice(boxDensity, sliceAxes3, slicePoint3, splitOrder)
    val leaves3 = density3.densityMap.truncation.leaves
    val values3 = density3.densityMap.vals
    assert(leaves3.length == 1)
    assert(leaves3(0) == NodeLabel(2))
    assert(values3(0) == (0.5, 1))

    val sliceAxes4 = Vector(2)
    val slicePoint4 = Vector(1.5)
    val density4 = quickSlice(boxDensity, sliceAxes4, slicePoint4, splitOrder)
    val leaves4 = density4.densityMap.truncation.leaves
    val values4 = density4.densityMap.vals
    assert(leaves4.length == 2)
    assert(leaves4(0) == NodeLabel(4))
    assert(leaves4(1) == NodeLabel(7))
    assert(values4(0) == (0.5, 1))
    assert(values4(1) == (0.5, 1))
  }
  
  "sample" should "produce samples from non-zero propability regions" in {

    val rng : UniformRandomProvider = RandomSource.XO_RO_SHI_RO_128_PP.create()

    val probabilities : Array[Double] = density.densityMap.vals.map(_._1).toArray
    val sampleSize = 1000000
    val min = -10.0
    val max = 10.0
    val sample = (density.sample(rng, sampleSize))

    val totProb = density.densityMap.vals.map(t => t._1 * t._2).reduce(_+_)
    assert(-0.00000001 < 1.0 - totProb && 1.0 - totProb < 0.00000001)

    for (i <- 0 until sampleSize) {
      assert(density.density(Vectors.dense(sample(i).toArray)) > 0.0)

      val unif = rng.nextDouble(min, max)
      assert(min <= unif && unif < max)
    }
  }

  "binarySearch" should "find first true value" in {
    assert(binarySearch((x : Int) => x >= 3)(Vector(0, 1, 2, 3, 3, 4, 5)) === 3)
  }

  "Rectangle" should "preserve total volume when split" in {
    for(i <- 0 until dfdim)
      assert(abs(bb.volume-bb.lower(i).volume-bb.upper(i).volume) < 0.0000001)
  }

  it should "split into equal volumes" in {
    for(i <- 0 until dfdim)
      assert(abs(bb.lower(i).volume - bb.upper(i).volume) < 0.0000001)
  }

  it should "have expected dimension" in {
    assert(bb.dimension === dfdim)
  }

  "single point" should "have volume 0" in {
    assert(point(df.first()).volume === 0)
  }

  "boundingBox" should "contain all points" in {
    val bb = boundingBox(df)
    assert(df.map(bb.contains(_)).reduce(_ && _))
  }

  "root node" should "have depth 0" in {
    assert(rootLabel.depth === 0)
  }

  "node" should "have its left/right leaf on the left/right" in {
    val node = NodeLabel(49)
    assert(node.left.isLeft)
    assert(node.right.isRight)
  }

  it should "be parent to its children" in {
    val node = NodeLabel(38)
    assert(node.children.forall(_.parent == node))
  }

  it should "have an ancestry consisting exactly of its ancestors" in {
    val node = NodeLabel(232)
    val ancestry1 = node.ancestors.toSet
    val ancestry2 = (1 until 232).
      map(NodeLabel(_)).
      filter(isAncestorOf(_, node)).
      toSet
    assert(ancestry1 === ancestry2)
  }

  // "new left/right ordering" should "agree with the old one" in {
  //   def isAncestorOfOld(a : NodeLabel, b : NodeLabel) : Boolean =
  //     a.lab < b.lab && (b.ancestor(b.depth - a.depth) == a)

  //   def isLeftOfOld(a : NodeLabel, b : NodeLabel) : Boolean =
  //     a.truncate(b.depth).lab < b.truncate(a.depth).lab

  //   val l = rootLabel.left
  //   val r = rootLabel.right
  //   val lll = l.left.left

  //   assert(isAncestorOfOld(l, lll) === isAncestorOf(l, lll))
  //   assert(isAncestorOfOld(l, r) === isAncestorOf(l, r))
  //   assert(isLeftOfOld(l, lll) === isLeftOf(l, lll))
  //   assert(isLeftOfOld(l, r) === isLeftOf(l, r))
  // }

  "MRSName" should "work on test cases" in {
    val lrllr = rootLabel.left.right.left.left.right
    assert(rootLabel.mrsName === "X")
    assert(lrllr.mrsName === "XLRLLR")
  }

  "join" should "be a supremum" in {
    val j = rootLabel.left.right.right
    val a = j.left.right.right.left.right.right
    val b = j.right.right.left.left.right
    assert(j == join(a, b))
    assert(a == join(a, a))
    assert(b == join(b, b))
    assert(j == join(j, a))
    assert(j == join(j, b))
  }

  "lefts" should "give the path from the root node" in {
    val lab = rootLabel.left.left.right.left
    val expectedLefts = Vector(true, false, true, true, false)
    assert(lab.lefts.toVector === expectedLefts)
  }

  "rights" should "give the path from the root node" in {
    val lab = rootLabel.left.left.right.left
    val expectedLefts = Vector(false, true, false, false, true)
    assert(lab.rights.toVector === expectedLefts)
  }

  "initialLefts" should "give number of initial left steps" in {
    val lab1 = rootLabel.left.left.right
    val lab2 = rootLabel.left.left.left.right.left
    assert(rootLabel.initialLefts === 0)
    assert(lab1.initialLefts === 2)
    assert(lab2.initialLefts === 3)
  }

  "initialRights" should "give number of initial right steps" in {
    val lab1 = rootLabel.right.right.left
    val lab2 = rootLabel.right.right.right.left.right
    assert(rootLabel.initialRights === 0)
    assert(lab1.initialRights === 2)
    assert(lab2.initialRights === 3)
  }

  // "leftmostAncestor" should "be youngest leftmost ancestor" in {
  //   val lab = rootLabel.left.left.right.right.left
  //   assert(isAncestorOf(lab.leftmostAncestor, lab))
  //   assert(!isAncestorOf(lab.leftmostAncestor.left, lab))
  // }

  // "rightmostAncestor" should "be youngest rightmost ancestor" in {
  //   val lab = rootLabel.right.right.left.left.right
  //   assert(isAncestorOf(lab.rightmostAncestor, lab))
  //   assert(!isAncestorOf(lab.rightmostAncestor.right, lab))
  // }

  "path" should "be a sorted open interval" in {
    val start = rootLabel.left.right.left.left.left.right
    val stop  = rootLabel.right.left.left.right
    val p = path(start, stop)
    p.sliding(2).foreach(x => assert(adjacent(x(0), x(1))))
    assert(adjacent(p.head, start))
    assert(adjacent(p.last, stop))
    assert(path(start, start).isEmpty)
    assert(path(stop, stop).isEmpty)
  }

  it should "be a walk" in {
    val start = rootLabel.left.right.left.left.left.right
    val stop  = rootLabel.right.left.left.right
    val p = path(start, stop)
    p.sliding(2).forall {
      case x => adjacent(x(0), x(1))
    }
  }

  "truncation" should "have size 1 when truncated at root" in {
    assert(rootTruncation.allNodes.size === 1)
  }

  it should "have size 1 at a leaf" in {
    val trunc = fromLeafSet(Set(rootLabel.left, rootLabel.right))
    assert(trunc.subtree(rootLabel.left) === Subset(0,1))
    assert(trunc.subtree(rootLabel.right) === Subset(1,2))
  }

  it should "have leaves in left/right order" in {
    val trunc = fromLeafSet(Set(rootLabel.right, rootLabel.left))
    isLeftOf(trunc.leaves(0), trunc.leaves(1))
  }

  it should "have completion in left/right order" in {
    // def lims(tv : Volume, tc : Count)(d : Int, v : Volume, c : Count) : Boolean =
    //   c > 100 || (1 - c/tc)*v/tv > 0.1

    // val h = histogram(df, lims, noEarlyStop)

    h.counts.minimalCompletionNodes.sliding(2).foreach {
      case ((llab, _) #:: (rlab, _) #:: _) =>
        isLeftOf(llab, rlab)
    }
  }

  it should "have completion be a finite tree" in {
    // TODO: Something is wrong with this test, didn't catch a bad bug!

    val leaves1 = Set(rootLabel.left.left.right, rootLabel.right.left)
    val trunc = fromLeafSet(leaves1)
    val leaves = trunc.minimalCompletion.leaves.toSet
    // print("Start: ")
    // println(leaves1)
    // print("Completed: ")
    // println(leaves)

    leaves.foreach {
      case l =>
        assert(!leaves.exists(isAncestorOf(_,l)))
    }

    val d = leaves.map(_.depth).max + 1
    var nodes = Set(rootLabel)
    for(i <- 1 to d) {
      // print("N: ")
      // println(nodes)
      nodes = (nodes -- leaves).flatMap(x => Set(x.left, x.right))
    }
    assert(nodes === Set())
  }

  "unfoldTree" should "interprets nodelabels correctly" in {
    val t = unfoldTree(rootLabel)((_, lab) => lab.left, (_, lab) => lab.right)(_)
    List(rootLabel, rootLabel.left, rootLabel.right, rootLabel.left.left, rootLabel.right.left).foreach {
      case lab =>
        assert(t(lab) === lab)
    }
  }

  "cached unfold" should "should agree with uncached" in {
    val l  = rootLabel.left
    val rl = rootLabel.right.left
    val rr = rootLabel.right.right
    val ll = rootLabel.left.left
    val lr = rootLabel.left.right
    val c  = tree.cellAtCached.recache(Set(l, rl, rr))
    for(lab <- List(rootLabel, l, rl, rr, ll, lr)) {
      assert(c(lab).low === tree.cellAt(lab).low)
      assert(c(lab).high === tree.cellAt(lab).high)
    }
  }

  "leafMap" should "agree with Map on values" in {
    val l = rootLabel.left
    val r = rootLabel.right
    val m1 = Map(l -> 1, r -> 2)
    val m2 = fromNodeLabelMap(m1)
    assert(m2.query(rootLabel #:: l #:: Stream.empty)._2 === m1.get(l))
    assert(m2.query(rootLabel #:: r #:: Stream.empty)._2 === m1.get(r))
  }

  it should "have leaves in left/right order" in {
    val l = rootLabel.left
    val r = rootLabel.right
    val m1 = Map(l -> 1, r -> 2)
    val m2 = fromNodeLabelMap(m1)
    isLeftOf(m2.truncation.leaves(0), m2.truncation.leaves(1))
  }

  it should "have toMap as inverse to fromNodeLabelMap" in {
    val l = rootLabel.left
    val r = rootLabel.right
    val m1 = Map(l -> 1, r -> 2)
    val m2 = fromNodeLabelMap(m1)
    assert(m1 === m2.toMap)
  }

  "spatialTree" should "have bounding box at root" in {
    assert(tree.cellAt(rootLabel) === bb)
  }

  it should "first split on first coordinate" in {
    assert(tree.axisAt(rootLabel) === 0)
  }

  it should "actually split along the right axes" in {
    assert(tree.cellAt(rootLabel.left).centre(0) != tree.cellAt(rootLabel).centre(0))
    assert(tree.cellAt(rootLabel.left.left).centre(1) != tree.cellAt(rootLabel.left).centre(1))
    assert(tree.cellAt(rootLabel.left.left.left).centre(2) != tree.cellAt(rootLabel.left.left).centre(2))
  }

  "descendSpatialTree" should "always terminate" in {
    val trunc = rootTruncation
    val x = df.takeSample(true, 1).head
    assert(rootLabel === trunc.descendUntilLeaf(tree.descendBox(x)))
  }

  "descendUntilLeaf" should "end in truncation" in {
    val t = h.counts.truncation
    val s = h.tree
    dfLocal.foreach(x => assert(t.leaves.contains(t.descendUntilLeaf(s.descendBox(x)))))
  }

  "descendBox" should "remain in cell containing point" in {
    val x = df.takeSample(true, 1).head
    val walk = tree.descendBox(x).take(10)
    var i = 0
    walk.foreach {
      case lab =>
        val cell = tree.cellAt(lab)
        assert(cell.contains(x))
    }
  }

  it should "have boxes aggreing with cellAt" in {
    val x = df.takeSample(true, 1).head
    val walk = tree.descendBoxPrime(x).take(10)
    walk.foreach {
      case (lab, box1) =>
        val box2 = tree.cellAt(lab)
        assert(box1.low === box2.low)
        assert(box1.high === box2.high)
    }
  }

  def assertdistinct[A](a : Seq[A]) : Unit = {
    a.tails.foreach {
      case Seq(x, xs@_*) => xs.foreach {
        case y => assert(y != x)
      }
      case Seq() => ()
    }
  }

  "cherries" should "all be cherries" in {
    h.counts.cherries(_+_).foreach {
      case (lab, _) => assert(h.counts.truncation.hasAsCherry(lab))
    }
  }

  it should "give distinct ones" in {
    // println(h.counts.cherries(_+_).map(_._1).toVector)
    assertdistinct(h.counts.cherries(_+_).map(_._1).toSeq)
  }

  "splitAndCountFrom" should "have only non-splittable boxes and splittable parents" in {
    val counts = splitAndCountFrom(tree, rootTruncation, df, lims, noEarlyStop)

    for((l, c) <- counts) {
      assert(!lims(tree.volumeAt(rootLabel), counts.values.sum)(l.depth, tree.volumeAt(l), c))
    }

    val parents : Map[NodeLabel, Count] = counts.keySet.map(_.parent).map(x => (x, counts getOrElse (x, 0L))).toMap
    for((l, c) <- parents) {
      assert(lims(tree.volumeAt(rootLabel), counts.values.sum)(l.depth, tree.volumeAt(l), c))
    }
  }

  it should "compute correct cell counts" in {
    def lims(tv : Volume, tc : Count)(d : Int, v : Volume, c : Count) : Boolean =
      c > 100 || (1 - c/tc)*v/tv > 0.1

    val counts = splitAndCountFrom(tree, rootTruncation, df, lims, noEarlyStop)

    for((l, c) <- counts) {
      val b = tree.cellAt(l)
      assert(c === df.filter(b.contains(_)).count)
    }
  }

  "internal" should "finds all internal nodes" in {
    val t = fromNodeLabelMap(List(rootLabel.left.left.right,
                                  rootLabel.left.right.right,
                                  rootLabel.right.right.left).map((_, ())).toMap)
    val internals1 = (BigInt(1) to t.truncation.leaves.map(_.lab).max).
      map(NodeLabel(_)).
      filter(x => t.truncation.leaves.exists(isAncestorOf(x, _))).
      toSet
    val internals2 = t.
      internal((), (_ : Unit, _ : Unit) => ()).
      map(_._1).
      toSet
    assert(internals1 === internals2)
  }

  it should "accumulate correct values" in {
    val t = fromNodeLabelMap(List(rootLabel.left.left.right,
                                  rootLabel.left.right.right,
                                  rootLabel.right.right.left).map(x => (x, Set(x))).toMap)

    val internals1 = (BigInt(1) to t.truncation.leaves.map(_.lab).max).
      map(NodeLabel(_)).
      filter(x => t.truncation.leaves.exists(isAncestorOf(x, _))).
      map(lab => (lab, t.slice(t.truncation.subtree(lab)).reduce(_.union(_)))).
      toSet

    val internals2 = t.
      internal(Set.empty, _.union(_)).
      toSet

    assert(internals1 === internals2)
  }

  "backtrack" should "traverses all ancestors in correct order" in {
    // def prio(lab : NodeLabel, c : Count, v : Volume) : Count = c
    def prio(lab : NodeLabel, c : Count, v : Volume) : Double =
      (1 - (1.0*c)/h.totalCount)*v

    // def lims(tv : Volume, tc : Count)(d : Int, v : Volume, c : Count) : Boolean =
    //   c > 100 || (1 - c/tc)*v/tv > 0.1
    // val h = histogram(df, lims, noEarlyStop)

    // def go(xs : Stream[(NodeLabel, Count)]) : Boolean = cs match {
    //   case Stream.Empty => true
    //   case ((lab, c) #:: xss) => (vs.forall { case (lab2, c2) => isAncestorOf(lab2, lab) }) && go(xss)
    // }
    // val nrinternals = (BigInt(1) to h.counts.truncation.leaves.map(_.lab).max).
    //   map(NodeLabel(_)).
    //   filter(x => h.counts.truncation.leaves.exists(isAncestorOf(x, _))).
    //   size

    val (splits, _ #:: hs) = h.backtrackWithNodes(prio)
    val bt = (splits zip hs).map { case ((a, b), c) => (a, b, c) }

    bt.tails.foreach {
      case Stream.Empty => ()
      case ((_, lab1, h1) #:: rest) =>
        rest.foreach {
          case (_, lab2, _) => assert(!isAncestorOf(lab1, lab2))
        }
        // rest.foreach {
        //   case (lab2, h2) => assert(isAncestorOf(lab2, lab1) || c1 <= c2)
        // }
        rest.exists {
          case (_, lab2, _) => lab2 === lab1.parent
        }
    }

    val (mergeOrder, hist) = h.backtrackNumStepsVerification(prio, 16*h.counts.vals.length)
    var i = 0

    bt.tails.foreach {
      case Stream.Empty => ()
      case ((_, lab1, h1) #:: rest) => {
        assert(lab1 == mergeOrder(i)._2)
        rest.foreach {
          case (_, lab2, _) => {
            assert(!isAncestorOf(lab1, lab2))
          }
        }
        // rest.foreach {
        //   case (lab2, h2) => assert(isAncestorOf(lab2, lab1) || c1 <= c2)
        // }
        if (!rest.isEmpty) {
          assert(rest.exists {
            case (_, lab2, _) => lab2 === lab1.parent
          })
        }
        if (!mergeOrder.drop(i+1).isEmpty) {
          assert(mergeOrder.drop(i+1).exists {
            case (_, lab2) =>  lab2 === lab1.parent
          })
        }

        i += 1
      }
    }
  }

  it should "only traverse ancestors" in {
    // def prio(lab : NodeLabel, c : Count, v : Volume) : Count = c
    def prio(lab : NodeLabel, c : Count, v : Volume) : Double =
      (1 - (1.0*c)/h.totalCount)*v
    
    val (mergeOrder, hist) = h.backtrackNumStepsVerification(prio, 16*h.counts.vals.length)
    mergeOrder.map(t => t._2).foreach(node => {
      assert(h.counts.truncation.leaves.exists {
        case leaf => isAncestorOf(node, leaf)
      })
    })

    h.backtrackNodes(prio).foreach {
      case n =>
        h.counts.truncation.leaves.exists {
          case n2 =>
            isAncestorOf(n, n2)
        }
    }
  }

  it should "traverse everything once" in {
    // def prio(lab : NodeLabel, c : Count, v : Volume) : Count = c
    def prio(lab : NodeLabel, c : Count, v : Volume) : Double =
      (1 - (1.0*c)/h.totalCount)*v
    val tracked = h.backtrackNodes(prio).toVector.toStream
    assertdistinct(tracked)

    val (mergeOrder, hist) = h.backtrackNumStepsVerification(prio, 16*h.counts.vals.length)
    assertdistinct(mergeOrder)
  }

  it should "begin/end in starting/trivial histogram" in {
    // def prio(lab : NodeLabel, c : Count, v : Volume) : Count = c
    def prio(lab : NodeLabel, c : Count, v : Volume) : Double =
      (1 - (1.0*c)/h.totalCount)*v

    assert(h.backtrack(prio).head === h)
    assert(h.backtrack(prio).last.counts.truncation.leaves === Vector(rootLabel))

    val (mergeOrder, hist) = h.backtrackNumStepsVerification(prio, 16*h.counts.vals.length)
    assert(hist.counts.truncation.leaves == Vector(rootLabel))
    assert(hist.counts.vals == Vector(h.totalCount))
    assert(hist.totalCount == h.totalCount)
  }

  it should "traverse all ancestors" in {
    // def prio(lab : NodeLabel, c : Count, v : Volume) : Count = c
    def prio(lab : NodeLabel, c : Count, v : Volume) : Double =
      (1 - (1.0*c)/h.totalCount)*v

    // h.backtrack(prio).toVector.reverse.take(10).foreach(x=>println(x.counts.truncation.leaves))

    val tracked = h.backtrackNodes(prio).toSet
    val full = h.counts.truncation.leaves.toSet.flatMap((x : NodeLabel) => x.ancestors())
    assert((tracked -- full).isEmpty)
    assert((full -- tracked).isEmpty)

    val (mergeOrder, hist) = h.backtrackNumStepsVerification(prio, 16*h.counts.vals.length)
    val traversed = mergeOrder.map(_._2).toSet
    assert((tracked -- traversed).isEmpty)
    assert((traversed -- tracked).isEmpty)

    // h.counts.truncation.leaves.foreach {
    //   case c =>
    //     c.ancestors.foreach {
    //       case a =>
    //         assert(tracked(a))
    //     }
    // }
  }

  it should "remove the reported leaf node" in {
    // def prio(lab : NodeLabel, c : Count, v : Volume) : Count = c
    def prio(lab : NodeLabel, c : Count, v : Volume) : Double =
      (1 - (1.0*c)/h.totalCount)*v

    val (splits, hs) = h.backtrackWithNodes(prio)

    (splits zip hs).foreach {
      case ((_,lab), hprev) =>
        val leaves = hprev.counts.truncation.leaves
        assert(!leaves.contains(lab))
        assert(leaves.contains(lab.left) || leaves.contains(lab.right))
    }

    (splits zip (hs.tail)).foreach {
      case ((_,lab), hprev) =>
        val leaves = hprev.counts.truncation.leaves
        assert(hprev.counts.truncation.leaves.contains(lab))
    }

    val (mergeOrder, hist) = h.backtrackNumStepsVerification(prio, 16*h.counts.vals.length)
    val hists : Array[Histogram] = new Array(mergeOrder.length + 1)
    hists(0) = h

    for (i <- 1 to mergeOrder.length) {
      val (arr, hist) = hists(i-1).backtrackNumStepsVerification(prio, 1)
      val n = arr(0)
      val leaves = hists(i-1).counts.truncation.leaves
      hists(i) = hist
      assert(!leaves.contains(n._2))
      assert(hists(i).counts.truncation.leaves.contains(n._2))
      assert(leaves.contains(n._2.left) || leaves.contains(n._2.right))
    }
  }

  it should "give histograms of decreasing size" in {
    // def prio(lab : NodeLabel, c : Count, v : Volume) : Count = c
    def prio(lab : NodeLabel, c : Count, v : Volume) : Double =
      (1 - (1.0*c)/h.totalCount)*v

    val (mergeOrder, hist) = h.backtrackNumStepsVerification(prio, 16*h.counts.vals.length)
    val hists : Array[Histogram] = new Array(mergeOrder.length + 1)
    hists(0) = h

    for (i <- 1 to mergeOrder.length) {
      val (arr, hist) = hists(i-1).backtrackNumStepsVerification(prio, 1)
      val leaves = hists(i-1).counts.truncation.leaves
      hists(i) = hist

      val diff1 = hists(i-1).truncation.leaves.toSet -- hists(i).truncation.leaves.toSet
      val diff2 = hists(i).truncation.leaves.toSet -- hists(i-1).truncation.leaves.toSet

      assert(diff1.size == 1 || diff1.size == 2)
      assert(diff2.size == 1)
      assert(diff1.forall(x => isAncestorOf(diff2.toVector(0), x)))
    }


    // def go(xs : Stream[(NodeLabel, Count)]) : Boolean = cs match {
    //   case Stream.Empty => true
    //   case ((lab, c) #:: xss) => (vs.forall { case (lab2, c2) => isAncestorOf(lab2, lab) }) && go(xss)
    // }
    // h.backtrack(prio).foreach {
    //   case x =>
    //     print("Backtrack Result: ")
    //     println(x.truncation.leaves.toList)
    // }
    // h.backtrack(prio).toStream.sliding(2).zip(h.backtrackNodes(prio).toStream.tail).foreach {
    h.backtrack(prio).toStream.sliding(2).foreach {
      case (h1 #:: h2 #:: rest) =>
        // print(h1.truncation.leaves.toList)
        // print("--")
        // println(h2.truncation.leaves.toList)
        val diff1 = h1.truncation.leaves.toSet -- h2.truncation.leaves.toSet
        val diff2 = h2.truncation.leaves.toSet -- h1.truncation.leaves.toSet
        // println(diff1)
        // println(diff2)

        // if(diff1.size != 1 && diff1.size != 2) {
        //   println("----------------------")
        //   println(h1.counts.truncation.subtree(lab))
        //   println("----------------------")
        //   println(h1.counts.truncation.leaves.filter(isAncestorOf(lab, _)))
        //   println("--------------------!!")
        //   println(h.counts.truncation.leaves.filter(isAncestorOf(lab, _)))
        //   println("----------------------")
        //   println(lab)
        //   println("----------------------")
        //   println(diff1)
        //   println("----------------------")
        //   println(diff2)
        //   println("----------------------")
        //   println(h1)
        //   println("----------------------")
        //   println(h2)
        //   println("----------------------")
        // }

        assert(diff1.size === 1 || diff1.size === 2)
        assert(diff2.size === 1)
        assert(diff1.forall(x => isAncestorOf(diff2.toVector(0), x)))
        // assert(diff2.forall(x => diff1.exists(y => isAncestorOf(x, y))))
        // // Extra sanity check, should be same as above
        // assert(h1.truncation.leaves.size <= h2.truncation.leaves.size + 1)
        // assert(h1.ncells <= h2.ncells + 1)
    }
  }

  "backtrackNumSteps" should "produce the same histogram as backtrack" in {
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd
    val spark = getSpark
    import spark.implicits._
   
    def prio(lab : NodeLabel, c : Count, v : Volume) : Double =
      (1 - (1.0*c)/h.totalCount)*v

    val dimensions = 10
    val sizeExp = 4

    val numPartitions = 4
    
    val trainSize = math.pow(10, sizeExp).toLong

    val rawTrainRDD = normalVectorRDD(sc, trainSize, dimensions, numPartitions, 1234567).cache

    var rootBox = RectangleFunctions.boundingBox(rawTrainRDD)

    val tree = widestSideTreeRootedAt(rootBox)
    val depth = 70

    println("Starting Regression Test on new backtrack routine")
    val labels = quickToLabeled(tree, depth, rawTrainRDD)
    val merged = mergeLeaves(tree, labels.toDS(), 20, depth / 8, "../tmp", true).collect.sortBy(_._1)(leftRightOrd)
    val hist = Histogram(tree, merged.map(_._2).reduce(_+_), fromNodeLabelMap(merged.toMap))

    var corr : Histogram = null
    val str = hist.backtrack(prio).zipWithIndex.drop(1).filter{ case (_, i) => i % 500 == 0}.takeWhile(_._2 <= 500).map(_._1).take(500)
    str.foreach(t => corr = t )
    val bt = hist.backtrackNumSteps(prio, 500)

    assert(bt.totalCount == corr.totalCount)
    assert(bt.counts.vals.length == bt.counts.truncation.leaves.length)
    assert(bt.counts.truncation.leaves.length == corr.counts.truncation.leaves.length)
    for (i <- 0 until bt.counts.vals.length) {
      assert(bt.counts.vals(i) == corr.counts.vals(i))
      assert(bt.counts.truncation.leaves(i) == corr.counts.truncation.leaves(i))
    }
  }

  "mergeSubtree" should "contain the correct leaves" in {
    val t = fromNodeLabelMap(Map(rootLabel.left.left.left    -> 1,
                                 rootLabel.left.left.right   -> 2,
                                 rootLabel.left.right.left   -> 3,
                                 rootLabel.left.right.right  -> 4,
                                 rootLabel.right.left.left   -> 5,
                                 rootLabel.right.left.right  -> 6,
                                 rootLabel.right.right.left  -> 7,
                                 rootLabel.right.right.right -> 8 ))
    val l = rootLabel.left
    val m = t.mergeSubtree(l, _ + _)

    val n = m.leaves.toSet -- t.leaves.toSet
    val d = t.leaves.toSet -- m.leaves.toSet

    assert(n == Set(l))
    d.foreach(x => assert(isAncestorOf(l, x)))
  }

  it should "contain the merged node" in {
    val t = fromNodeLabelMap(Map(rootLabel.left.left.left    -> 1,
                                 rootLabel.left.left.right   -> 2,
                                 rootLabel.left.right.left   -> 3,
                                 rootLabel.left.right.right  -> 4,
                                 rootLabel.right.left.left   -> 5,
                                 rootLabel.right.left.right  -> 6,
                                 rootLabel.right.right.left  -> 7,
                                 rootLabel.right.right.right -> 8 ))
    val l = rootLabel.left
    val m = t.mergeSubtree(l, _ + _)
    assert(!t.truncation.leaves.contains(l))
    assert(m.truncation.leaves.contains(l))
  }

  "tailProbabilities" should "should be probabilities, attaining maximum 1" in {
    val tp = h.tailProbabilities()
    tp.tails.vals.foreach {
      case p =>
        assert(0 <= p)
        assert(p <= 1)
    }
    assert(tp.tails.vals.max === 1)
  }

  it should "have tail 0/1 in minimum/maximum density cell" in {
    val tp   = h.tailProbabilities()
    val imin = tp.tails.vals.zipWithIndex.maxBy(_._1)._2
    val imax = tp.tails.vals.zipWithIndex.minBy(_._1)._2
    val cmax = h.counts.vals(imax)
    val cmin = h.counts.vals(imin)
    assert(cmax === h.counts.vals.max)
    assert(cmin === h.counts.vals.min)
  }

  it should "produce increasingly larger coverage regions" in {
    val spark = getSpark
    import spark.implicits._
    implicit val ordering : Ordering[NodeLabel] = leftRightOrd
   
    val dimensions = 3
    val sizeExp = 5

    val numPartitions = 16
    
    val trainSize = math.pow(10, sizeExp).toLong
    val finestResSideLength = 1e-1

    val rawTrainRDD = normalVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions, 1234567)
    val rawTestRDD =  normalVectorRDD(spark.sparkContext, trainSize/2, dimensions, numPartitions, 7654321)

    var rectTrain = RectangleFunctions.boundingBox(rawTrainRDD)
    var rectTest = RectangleFunctions.boundingBox(rawTestRDD)
    //val rootBox = RectangleFunctions.hull(rectTrain, rectTest)
    val rootBox = Rectangle(Vector(-80.0, -80.0, -80.0), Vector(80,80,80))

    val tree = widestSideTreeRootedAt(rootBox)
    val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth
    val stepSize = 1500 
    val kInMDE = 10

    var countedTrain = quickToLabeled(tree, finestResDepth, rawTrainRDD)
    var countedTest = quickToLabeled(tree, finestResDepth, rawTestRDD)
        
    val partitioner = new SubtreePartitioner(2, countedTrain, 20) /* action 1 (collect) */
    val depthLimit = partitioner.maxSubtreeDepth
    val countLimit = 30
    val subtreeRDD = countedTrain.repartitionAndSortWithinPartitions(partitioner)
    val merged = mergeLeavesRDD(subtreeRDD, countLimit, depthLimit, true)

    val hist = Histogram(tree, merged.map(_._2).reduce(_+_), fromNodeLabelMap(merged.toMap))
    var stopSize = Option.empty[Int]
    
    density = toDensityHistogram(getMDE(
      hist,
      countedTest, 
      trainSize/2,
      kInMDE, 
      true 
    )).normalize

    val coverageRegions = density.tailProbabilities.tails.toMap
    val sorted = coverageRegions.iterator.toArray.sortBy(_._2)
    val vals = sorted.map(kv => kv._2)
    val keys = sorted.map(_._1)
    var sum = 0.0
    
    val wantedConfidence = 0.95
    var actualConfidence = 0.0

    for (i <- 0 until vals.length) {

      assert(vals(i) == coverageRegions(keys(i)))
      assert(sum < coverageRegions(keys(i)))
      val densityMap = density.densityMap.toMap
      val p1 = coverageRegions(keys(i)) - sum
      val p2 = densityMap.get(keys(i)) match {
        case None => 10.0 
        case Some(probVol) => probVol._1 * probVol._2
      }
      val diff = p1 - p2

      assert(-1e-10 < diff && diff < 1e-10)

      val box = density.tree.cellAt(keys(i))
      val middle = Vectors.dense(box.centre(0), box.centre(1), box.centre(2))
      assert(density.tailProbabilities.query(middle) == coverageRegions(keys(i)))

      sum = coverageRegions(keys(i))
      if (wantedConfidence <= sum && actualConfidence == 0.0) {
        actualConfidence = sum
      }
    }

    assert(density.tailProbabilities.confidenceRegion(wantedConfidence) == actualConfidence)

    val outsidePoint = Vectors.dense(100.0, 100.0, 100.0)
    assert(density.tailProbabilities.query(outsidePoint) == 1.0)

    var nullSetLeaf : NodeLabel = NodeLabel(64) /* (-80.0, -40.0) x (-80.0, -40.0) x (-80.0, -40.0) */
    

    val nullSetBox = density.tree.cellAt(nullSetLeaf) 
    assert(coverageRegions.get(nullSetLeaf) == None)
    val mid = Vectors.dense(nullSetBox.centre(0), nullSetBox.centre(1), nullSetBox.centre(2))
    val va = density.tailProbabilities.query(mid)
    assert(va == 1.0)
  }
  
  it should "Add largest density value regions first" in {
    val leaves = Truncation(Vector(NodeLabel(4), NodeLabel(5), NodeLabel(6), NodeLabel(7)))
    val densityMap = LeafMap(leaves, Vector((0.5, 1.0),(0.25, 1.0),(0.15, 1.0),(0.1, 1.0)))
    val tree = WidestSplitTree(Rectangle(Vector(-2.0), Vector(2.0)))
    val dens = DensityHistogram(tree, densityMap)
    val coverageRegions = dens.tailProbabilities.tails.toMap

    var leaf = 4
    var box = dens.tree.cellAt(NodeLabel(leaf))
    var middle = Vectors.dense(box.centre(0))
    assert(dens.tailProbabilities.query(middle) == 0.5)

    leaf = 5 
    box = dens.tree.cellAt(NodeLabel(leaf))
    middle = Vectors.dense(box.centre(0))
    assert(dens.tailProbabilities.query(middle) == 0.75)

    leaf = 6 
    box = dens.tree.cellAt(NodeLabel(leaf))
    middle = Vectors.dense(box.centre(0))
    assert(dens.tailProbabilities.query(middle) == 0.90)

    leaf = 7 
    box = dens.tree.cellAt(NodeLabel(leaf))
    middle = Vectors.dense(box.centre(0))
    assert(dens.tailProbabilities.query(middle) == 1.0)

    assert(dens.tailProbabilities.confidenceRegion(0.0) == 0.5)
    assert(dens.tailProbabilities.confidenceRegion(0.5) == 0.5)
    assert(dens.tailProbabilities.confidenceRegion(0.74) == 0.75)
    assert(dens.tailProbabilities.confidenceRegion(0.75) == 0.75)
    assert(dens.tailProbabilities.confidenceRegion(0.89) == 0.90)
    assert(dens.tailProbabilities.confidenceRegion(0.90) == 0.90)
    assert(dens.tailProbabilities.confidenceRegion(0.91) == 1.0)
    assert(dens.tailProbabilities.confidenceRegion(1.00) == 1.0)
  }



  "fringes" should "have inverse concatLeafmap" in {
    val parentTrunc = h.counts.truncation match {
      case Truncation(leaves) => Truncation(leaves.map(_.parent).distinct)
    }
    assert(concatLeafMaps(fringes(h.counts, parentTrunc).vals).vals == h.counts.vals)
  }

  it should "have subtree at each key" in {
    val parentTrunc = h.counts.truncation match {
      case Truncation(leaves) => Truncation(leaves.map(_.parent).distinct)
    }
    val fs = fringes(h.counts, parentTrunc)
    fs.truncation.leaves.zip(fs.vals).foreach {
      case (p, LeafMap(Truncation(ls), _)) =>
        assert(ls.forall(isAncestorOf(p, _)))
    }
  }

  "backtrackTo" should "behave like backtrack when target is root" in {
    // def prio(lab : NodeLabel, c : Count, v : Volume) : Count = c
    def prio(lab : NodeLabel, c : Count, v : Volume) : Double =
      (1 - (1.0*c)/h.totalCount)*v

    val (splits, _ #:: hs) = h.backtrackWithNodes(prio)
    val bt = (splits zip hs).map { case ((a, b), c) => (a, b, c) }

    val (splitsT, _ #:: hsT) = h.backtrackToWithNodes(prio,bt.last._3)
    val btT = (splitsT zip hsT).map { case ((a, b), c) => (a, b, c) }

    assert(bt.length === btT.length)
    btT.zip(bt).foreach {
      case ((p1, lab1, h1), (p2, lab2, h2)) =>
        assert(p1 === p2)
        assert(lab1 === lab2)
        assert(h1.counts.truncation.leaves === h2.counts.truncation.leaves)
        assert(h1.counts.vals === h2.counts.vals)
    }
  }

  it should "be an coarsing sequence of valid histograms" in {
    def prio(lab : NodeLabel, c : Count, v : Volume) : Double = (1 - (1.0*c)/h.totalCount)*v
    def prioC(lab : NodeLabel, c : Count, v : Volume) : Count = c

    val (splits, _ #:: hs) = h.backtrackWithNodes(prio)
    val bt = (splits zip hs).map { case ((a, b), c) => (a, b, c) }

    assert(bt.length > 20)
    val interm = bt.takeRight(15).head._3

    val (splitsT, _ #:: hsT) = h.backtrackToWithNodes(prioC,interm)
    val btT = (splitsT zip hsT).map { case ((a, b), c) => (a, b, c) }

    btT.sliding(2).foreach {
      case Stream((_, _, h1), (_, _, h2)) =>
        assert(h1.counts.truncation.leaves === h1.counts.truncation.leaves.sorted(leftRightOrd))
        val diff1 = h1.counts.truncation.leaves.toSet -- h2.counts.truncation.leaves.toSet
        val diff2 = h2.counts.truncation.leaves.toSet -- h1.counts.truncation.leaves.toSet
        assert(diff1.size === 1 || diff1.size == 2)
        assert(diff2.size === 1)
        diff2.foreach {
          case p =>
          diff1.foreach {
            case c =>
              assert(p === c.parent)
          }
        }
    }
  }

  it should "begin/end in starting/target histogram" in {
    def prio(lab : NodeLabel, c : Count, v : Volume) : Double = (1 - (1.0*c)/h.totalCount)*v
    def prioC(lab : NodeLabel, c : Count, v : Volume) : Count = c

    val (splits, _ #:: hs) = h.backtrackWithNodes(prio)
    val bt = (splits zip hs).map { case ((a, b), c) => (a, b, c) }

    assert(bt.length > 20)
    val interm = bt.takeRight(15).head._3

    assert(h.backtrackTo(prioC,interm).head === h)
    assert(h.backtrackTo(prioC,interm).last === interm)
  }

  // it should "give refinements of the target" in {
  //   def prio(lab : NodeLabel, c : Count, v : Volume) : Double = (1 - (1.0*c)/h.totalCount)*v
  //   def prioC(lab : NodeLabel, c : Count, v : Volume) : Count = c

  //   val (splits, _ #:: hs) = h.backtrackWithNodes(prio)
  //   val bt = (splits zip hs).map { case ((a, b), c) => (a, b, c) }

  //   assert(bt.length > 20)
  //   val interm = bt.takeRight(15).head._3

  //   val (splitsT, _ #:: hsT) = h.backtrackToWithNodes(prioC,interm)
  //   val btT = (splitsT zip hsT).map { case ((a, b), c) => (a, b, c) }

  //   btT.zipWithIndex.foreach {
  //     case ((_, _, h), i) =>
  //       withClue (i.toString) {
  //       h.counts.truncation.leaves.foreach {
  //         case l1 =>
  //           assert(interm.counts.truncation.leaves.exists {
  //                    case l2 =>
  //                      l2 == l1 || isAncestorOf(l2, l1)
  //                  })
  //       }
  //       interm.counts.truncation.leaves.foreach {
  //         case l2 =>
  //           assert(h.counts.truncation.leaves.exists {
  //                    case l1 =>
  //                      l2 == l1 || isAncestorOf(l2, l1)
  //                  })
  //       }
  //       }
  //   }
  //   assert(h.backtrackTo(prioC,interm).last === interm)
  // }

  it should "behave like prefix of backtrack when target is intermediate node" in {
    // def prio(lab : NodeLabel, c : Count, v : Volume) : (Count, BigInt) = (c, lab.lab)
    def prio(lab : NodeLabel, c : Count, v : Volume) : (Double, BigInt) =
      ((1 - (1.0*c)/h.totalCount)*v, lab.lab)

    val (splits, _ #:: hs) = h.backtrackWithNodes(prio)
    val bt = (splits zip hs).map { case ((a, b), c) => (a, b, c) }

    assert(bt.length > 20)

    val (splitsT, _ #:: hsT) = h.backtrackToWithNodes(prio,bt.takeRight(15).head._3)
    val btT = (splitsT zip hsT).map { case ((a, b), c) => (a, b, c) }

    assert(bt.length === btT.length+15-1)
    btT.zip(bt).zipWithIndex.foreach {
      case (((_, _, h1), (_, _, h2)), i) =>
        val l1 = h1.counts.truncation.leaves.toSet
        val l2 = h2.counts.truncation.leaves.toSet
        assert((i, l1 -- l2, l2 -- l1) === (i, Set.empty, Set.empty))
        assert((i, h1.counts.vals) === (i, h2.counts.vals))
    }
    assert(btT.map(_._2) === bt.map(_._2).take(btT.length))
    assert(btT.map(_._1) === bt.map(_._1).take(btT.length))
  }

  "size of completion" should "should be decreasing" in {
    def prio(lab : NodeLabel, c : Count, v : Volume) : (Double, BigInt) =
      ((1 - (1.0*c)/h.totalCount)*v, lab.lab)
    h.backtrack(prio).toStream.map(_.counts.truncation.minimalCompletionNodes.size).sliding(2).foreach {
      case x => assert(x(0) === x(1) + 1)
    }
  }

  "histogramStartingWith" should "generate a refinement" in {
    val leaves1 = h.counts.truncation.leaves
    val leaves2 = h2.counts.truncation.leaves
    val bads = leaves2.filter(lab1 => !leaves1.exists(lab2 => lab2 == lab1 || isAncestorOf(lab2, lab1))).toSet
    assert(bads.isEmpty)
  }
  // it should "never backtrack beyond goal" in {
  //   def prio1(lab : NodeLabel, c : Count, v : Volume) : (Count, BigInt) = (c, lab.lab)
  //   def prio(lab : NodeLabel, c : Count, v : Volume) : Count =
  //     (1 - c/h.totalCount)*v
  // }

  // "backtrackTo" should "end in the target histogram" in {
  //   def supportCarveLim(totalVolume : Double, totalCount : Count)(depth : Int, volume : Volume, count : Count) =
  //     count > dfnum/2 || (1 - count/totalCount)*volume/totalVolume > 0.01

  //   def countLim(totalVolume : Double, totalCount : Count)(depth : Int, volume : Volume, count : Count) =
  //     count > 5

  //   // factor out totalCount and totalVolume since they are the same for all nodes
  //   def supportCarvePriority(lab : NodeLabel, c : Count, v : Volume) : Double = (1 - c)*v
  //   def countPriority(lab : NodeLabel, c : Count, v : Volume) : Count = c

  //   val supportCarvedH = histogram(df, supportCarveLim, noEarlyStop)
  //   val tributaryH = histogramStartingWith(supportCarvedH, df, countLim, noEarlyStop)
  //   val btT = tributaryH.backtrackToWithNodes(countPriority, supportCarvedH)
  //   // println(btT.map(_._3.counts.vals.take(4).toVector))
  //   val l = btT.toStream.last._3

  //   assert(l.truncation.leaves === supportCarvedH.truncation.leaves)
  //   assert(l.counts.vals === supportCarvedH.counts.vals)
  // }
}
