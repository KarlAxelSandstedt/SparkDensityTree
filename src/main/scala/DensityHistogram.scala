/**************************************************************************
 * Copyright 2022 Johannes Graner
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **************************************************************************/

package co.wiklund.disthist

import java.math.BigInteger
import org.apache.commons.rng.UniformRandomProvider;
import org.apache.commons.rng.sampling.distribution.SharedStateDiscreteSampler
import org.apache.commons.rng.sampling.distribution.SharedStateDiscreteSampler._
import org.apache.commons.rng.sampling.distribution.AliasMethodDiscreteSampler
import org.apache.commons.rng.sampling.distribution.AliasMethodDiscreteSampler._

import Types._
import LeafMapFunctions._
import SpatialTreeFunctions._
import NodeLabelFunctions._
import TruncationFunctions._
import HistogramFunctions._

import org.apache.spark.mllib.linalg.Vectors

/**
 * DensityHistogram - Histogram density class. The class keeps track of the underlying spatial partitioning and
 *                    contains a LeafMap mapping NodeLabels of leaves to a (densityValue, volume) tuple. The
 *                    densityValue is simply the value of the density takes on the whole space defined by a
 *                    leaf's cell.
 *
 *                    NOTE: The DensityHistogram is not necessarily a density, and the user must call .normalize
 *                    to generate a density which integrates to 1.
 *
 * @param tree - The DensityHistogram's root box
 * @param densityMap - Map from leaves to leaf count and leaf volume.
 */
case class DensityHistogram(tree: SpatialTree, densityMap: LeafMap[(Double, Volume)]) {

  /**
   * density - Determine the value of the density function at point v.
   *
   * @param v - The point at which we wish to determine the density.
   * @return The density at v.
   */
  def density(v: MLVector): Double = {
    val point = v.toArray
    for (i <- 0 until point.length) {
      if (point(i) < tree.rootCell.low(i) || point(i) > tree.rootCell.high(i)) {
        return 0.0
      }
    }

    densityMap.query(tree.descendBox(v))._2.getOrElse((0.0, 0.0))._1
  }

  /**
   * normalize - Construct a normalized density out of the DensityHistogram. 
   *
   * WARNING: The function is usually applied together with a slice operation to determine the conditional density.
   * Make sure that the slice is defined; In the case that the slice only slice 0-probability regions, this is
   * ill-defined, so check any returned quickSlice value if it equals null.
   *
   * @return The normalized density.
   */
  def normalize: DensityHistogram = {
    val integral = densityMap.vals.map{ case (dens, vol) => 
      dens * vol
    }.sum
    val normVals = densityMap.vals.map{ case (dens, vol) =>
      (dens / integral, vol)
    }
    copy(densityMap = densityMap.copy(vals = normVals))
    DensityHistogram(this.tree, this.densityMap.copy(vals = normVals))
  }

  /**
   * tailProbabilities - Constructs Map of increasingly larger coverage regions. Each coverage region is the union of the
   * previous and the next leaf whose density is the largest out of the set of leaves not yet inside a coverage region.
   *
   * NOTE: density / volume ratio may become very large for higher dimensions => Perhaps this will affect numerical stability?
   */
  def tailProbabilities() : TailProbabilities = {
    val quantiles = densityMap.toIterable.map {
      case (lab, densVol) => (lab, densVol._1, densVol._1 * densVol._2)
    }.toVector.sortBy {
      case (lab, dens, prob) => dens
    }.reverse.toIterable.scanLeft((rootLabel, 0.0)) {
      case ((_, probSum), (lab, _, prob)) => (lab, probSum + prob)
    }.tail.toMap

    TailProbabilities(tree, fromNodeLabelMap(quantiles))
  }

  /**
   * sample - Sample from the distribution defined by the density. The user must pass a random number generator form the Apache
   *          Commons RNG library.
   *
   * @param rng - Apache Commons RNG
   * @param sampleSize - Wanted size
   * @return A Sample of size sampleSize from the density histogram's distribution
   */
  def sample(rng : UniformRandomProvider, sampleSize : Int): Array[Vector[Double]] = {

    /* Sample which leaf to sample point from */
   val probabilities : Array[Double] = densityMap.vals.map(t => t._1 * t._2).toArray
    var indexSampler : SharedStateDiscreteSampler = AliasMethodDiscreteSampler.of(rng, probabilities)
  
    val boxes = densityMap.truncation.leaves.map(tree.cellAt(_))
    var sampleBuf : Array[Array[Double]] = Array.ofDim[Double](sampleSize, tree.dimension)

    for (i <- 0 until sampleSize) {
      val index = indexSampler.sample()
      for (j <- 0 until tree.dimension) {
        val box = boxes(index)
        sampleBuf(i)(j) = rng.nextDouble(box.low(j), box.high(j))
      }
    }

    sampleBuf.map(_.toVector)
  }
}

object HistogramFunctions {

  /**
   * toDensityHistogram - Generate a DensityHistogram from a Histogram
   *
   * @param hist - The Histogram to convert
   * @return The converted DensityHistogram
   */
  def toDensityHistogram(hist: Histogram): DensityHistogram = {
    val counts = hist.counts
    val tree = hist.tree
    val totalCount = hist.totalCount
    val totalVolume = tree.rootCell.volume

    val densitiesWithVolumes = counts.toIterable.map {
      case (node, c) => 
        val vol = tree.volumeAt(node)
        (node, (c / (totalCount * vol), vol))
    }.toMap

    DensityHistogram(tree, fromNodeLabelMap(densitiesWithVolumes))
  }
  
  /**
    * Removes axes from a rectangle
    *
    * @param rec
    * @param axesToKeep
    * @return
    */
  def marginalizeRectangle(rec: Rectangle, axesToKeep: Vector[Axis]): Rectangle = {
    Rectangle(axesToKeep map rec.low, axesToKeep map rec.high)
  }

  /**
    * The bit-value of the label as a vector
    * 
    * ex. 14 = b1110, so
    * getSplitDirections(NodeLabel(14)) = Vector(1,1,1,0)
    *
    * @param lab
    * @return
    */
  def getSplitDirections(lab: NodeLabel): Vector[Int] = {
    (0 to lab.depth - 1).toVector.map(i => 
      if (lab.lab.testBit(i)) 1 else 0
    )
  }

  def marginalize(hist: Histogram, axesToKeep: Vector[Axis]): DensityHistogram = {
    marginalize(toDensityHistogram(hist), axesToKeep)
  }

  def marginalize(densHist: DensityHistogram, axesToKeep: Vector[Axis]): DensityHistogram = {
    val densTree = densHist.tree
    val densAndVols = densHist.densityMap

    val splits = densTree.splits

    val marginalized = densAndVols.toIterable.groupBy{ case (lab, _) => 
      // group by boxes with unwanted axes removed
      marginalizeRectangle(densTree.cellAt(lab), axesToKeep)
    }.map{ case (newRec, densities) => 
      val newVol = newRec.volume
      // aggregate by sum of densities in the new box
      val newDensity = densities.map { case (_, (dens, vol)) =>
        dens * vol / newVol
      }.sum 
      // Find NodeLabel corresponding to new box
      val newLabel = {
        val oldLab = densities.head._1
        val axisSplits = splits.take(oldLab.depth).reverse
        val splitDirections = getSplitDirections(oldLab)
        val newDirections = axisSplits.zip(splitDirections).filter{ case (axis, _) =>
          axesToKeep.contains(axis)
        }.map{ case (_, direction) => direction }
        NodeLabel(newDirections.reverse.foldLeft(BigInt(1)){ case (labAcc, i) => (labAcc << 1) + i })
      } 
      (newRec, (newLabel, newDensity, newVol))
    }.map{ case(rec, (lab, dens, vol)) => (lab, (dens: Double, vol)) }

    val margTree = widestSideTreeRootedAt(marginalizeRectangle(densTree.rootCell, axesToKeep))

    // find the leaves in the marginalized tree,
    // and the ancestors grouped by depth
    val (margLeaves, margAncesAtDepth) = marginalized.keySet
      .groupBy(_.depth).toVector.sortBy(-_._1)
      .foldLeft((Set.empty[NodeLabel], Map.empty[Depth, Set[NodeLabel]])){
        case ((leafAcc, ancesAtDepths), (depth, nodesAtDepth)) =>
          val leavesAtDepth = leafAcc.map(_.truncate(depth))
          val newLeaves = nodesAtDepth -- leavesAtDepth
          val ancesAtDepth = depth -> (nodesAtDepth intersect leavesAtDepth)

          (leafAcc ++ newLeaves, ancesAtDepths + ancesAtDepth)
      }

    // find the missing leaves in the marginalized tree,
    // i.e. the ones which are not already leaves, but
    // have an ancestor in the keys of marginalization.
    val ancToDescsWithMissing = margAncesAtDepth.flatMap{
      case (depth, ancesAtDepth) =>

        // descendants to ancestors at the current depth
        val descs = margLeaves.filter(node => 
          ancesAtDepth.contains(node.truncate(depth))
        )

        // grouping descendants by their ancestor
        val ancToDescs = ancesAtDepth.map(anc => 
          anc -> descs.filter(desc => 
            isAncestorOf(anc, desc)
          )
        )

        // filling in the missing nodes by finding the minimal
        // completion of the subtree.
        ancToDescs.map{ case (anc, descs) =>
          val rootedDescs = descs.map( desc => 
            rootAtNode(anc, desc)
          )
          val rootedWithMissing = fromLeafSet(rootedDescs).minimalCompletion.leaves
          val descsWithMissing = rootedWithMissing.map(rootedDesc => 
            descendantFromRoot(anc, rootedDesc)
          )
          anc -> descsWithMissing
        }
    }

    // Map descendents to the sum of densities of ancestor boxes
    val descsWithDens = ancToDescsWithMissing.toVector
      .flatMap{ case (anc, descs) => 
        descs.map(desc => (desc, marginalized.getOrElse(anc, (0.0, 0.0))._1)) 
      }.groupBy{ case (lab, dens) => lab }
      .mapValues( labAndDens => 
        labAndDens.map{ case (lab, dens) => dens }.sum
      )

    // Add the densities from ancestor boxes to the descendants,
    // and remove all nodes that are not leaves.
    val margMap = descsWithDens.keys.foldLeft(marginalized){
      case (mapAcc, descLab) =>
        val oldValue = mapAcc.getOrElse(descLab, (0.0, margTree.volumeAt(descLab)))
        val newValue = (oldValue._1 + descsWithDens(descLab), oldValue._2)
        mapAcc.updated(descLab, newValue)
    }.filter{ case (lab, _) => !ancToDescsWithMissing.keySet.contains(lab) }

    DensityHistogram(margTree, fromNodeLabelMap(margMap))
  }

  @deprecated("Use faster quickSlice")
  def slice(densHist: DensityHistogram, sliceAxes: Vector[Axis], slicePoints: Vector[Double]): DensityHistogram = {
    slice(densHist, sliceAxes, Vectors.dense(slicePoints.toArray))
  }

  @deprecated("Use faster quickSlice")
  def slice(densHist: DensityHistogram, sliceAxes: Vector[Axis], slicePoints: MLVector): DensityHistogram = {
    
    val sliceBoxes = densHist.densityMap.leaves.map( node => 
      node -> marginalizeRectangle(densHist.tree.cellAt(node), sliceAxes)
    ).toMap.filter{ case (lab, rec) => 
      rec.contains(slicePoints)
    }

    val splits = densHist.tree.splits
    val nonSliceAxes = ((0 to densHist.tree.dimension - 1).toSet -- sliceAxes).toVector

    val newLabels = sliceBoxes.keys.toVector.map{ lab => 
      val axisSplits = splits.take(lab.depth).reverse
      val splitDirections = getSplitDirections(lab)

      val newDirections = axisSplits.zip(splitDirections)
        .filter{ case (axis, _) =>
          nonSliceAxes.contains(axis)
        }.map{ case (_, direction) => direction }

      val newLab = newDirections.reverse
        .foldLeft(BigInt(1)){ case (labAcc, i) => 
          (labAcc << 1) + i 
        }

      lab -> NodeLabel(newLab)
    }

    val densMap = densHist.densityMap.toMap
    val slicedRootBox = marginalizeRectangle(densHist.tree.rootCell, nonSliceAxes)
    val slicedTree = widestSideTreeRootedAt(slicedRootBox)

    val slicedNodeMap = newLabels.map{ case (oldLab, newLab) =>
      newLab -> (densMap(oldLab)._1, slicedTree.volumeAt(newLab))
    }.toMap

    DensityHistogram(slicedTree, fromNodeLabelMap(slicedNodeMap))
  }

  def quickSlice(densHist: DensityHistogram, sliceAxes: Vector[Axis], slicePoint: Vector[Double], splitOrder : Array[Axis], sliceLeavesBufC : Array[NodeLabel] = null, sliceValuesBufC : Array[(Double,Volume)] = null) : DensityHistogram = {
    quickSlice(densHist, sliceAxes, Vectors.dense(slicePoint.toArray), splitOrder, sliceLeavesBufC, sliceValuesBufC)
  }

  /**
   * quickSlice - Quick implementation of slice. Finds the conditional distribution of the density when conditioning on the 
   *              dimensions given in sliceAxes with their respective points set in slicePoint. For example:
   *              [Non-normalised] f(x_0 | X_1=x_1, X_2=x_2) => [sliceAxes=Vector(1,2), slicePoint=Vector(x1,x2)]
   *
   * @param densHist - Density Histogram to retrieve conditional density from
   * @param sliceAxes - Indices of dimensions to condition on
   * @param slicePoint - Point to condition on
   * @param splitOrder - Precalculated splitOrder. The array must contain enough splits for us to traverse down to the deepest leaf.
   *              It is thus suitable to simply calculate the splitOrder all the way down to the depth chosen to merge up from
   *              at the merging/backtracking stage of the construction of the histogram.
   * @param sliceLeavesBufC - Buffer to be used for conditional leaves, must be large enough to hold all conditional leaves
   * @param sliceValuesBufC - Buffer to be used for conditional leaf values must be large enough to hold all conditional leaf values
   * @return The conditional (non-normalised) histogram on If the slice point was found to be within at least one leaf. If no 
   *              no leaf was sliced, return null.
   */
  def quickSlice(densHist: DensityHistogram, sliceAxes: Vector[Axis], slicePoint: MLVector, splitOrder : Array[Axis], sliceLeavesBufC : Array[NodeLabel], sliceValuesBufC : Array[(Double,Volume)]) : DensityHistogram = {

    /* Check if slice point is within possible slice region */
    for (i <- 0 until sliceAxes.length) {
      if (slicePoint(i) < densHist.tree.rootCell.low(sliceAxes(i)) || densHist.tree.rootCell.high(sliceAxes(i)) < slicePoint(i)) {
        return null
      }
    }
    
    val nonSliceAxes = ((0 to densHist.tree.dimension - 1).toSet -- sliceAxes).toVector
    val slicedRootBox = marginalizeRectangle(densHist.tree.rootCell, nonSliceAxes)
    val slicedTree = widestSideTreeRootedAt(slicedRootBox)

    val leaves = densHist.densityMap.truncation.leaves
    val densMap = densHist.densityMap.toMap

    var sliceLeavesBuf = sliceLeavesBufC
    var sliceValuesBuf = sliceValuesBufC
    if (sliceLeavesBuf == null) { sliceLeavesBuf = new Array(leaves.length) }
    if (sliceValuesBuf == null) { sliceValuesBuf = new Array(leaves.length) }

    /* Reusable Array of box dimensions */
    var low : Array[Double] = densHist.tree.rootCell.low.toArray
    var high : Array[Double] = densHist.tree.rootCell.high.toArray

    /* extend slicePoint to full dimension for ease of use later */
    var slicePointExtended : Array[Double] = new Array(densHist.tree.rootCell.dimension)

    /* table of what axes need bit in new representation */
    var splitToBitNeeded : Array[Int] = new Array(densHist.tree.rootCell.dimension)

    var k = 0
    for (i <- 0 until splitToBitNeeded.length) {
      if (k < sliceAxes.length && i == sliceAxes(k)) {
        slicePointExtended(i) = slicePoint(k)
        splitToBitNeeded(sliceAxes(k)) = 0
        k += 1
      } else {
        splitToBitNeeded(i) = 1
        slicePointExtended(i) = 0.0
      }
    }

    /* old to new depth table */
    var oldToNewDepth : Array[Int] = new Array(splitOrder.length + 1)
    var newDepth = 0
    oldToNewDepth(0) = newDepth 

    for (i <- 1 until oldToNewDepth.length) {
      newDepth += splitToBitNeeded(splitOrder(i-1))
      oldToNewDepth(i) = newDepth
    }

    /* Get max depth and construct bit array which fit all possible new labels */
    val maxDepth = oldToNewDepth.last
    var numBits = maxDepth + 2
    val rest = numBits % 8 
    if (rest > 0) { 
       numBits += (8 - rest)
    }
    var bits : Array[Byte] = new Array(numBits / 8)

    var bufIndex = 0
    for (i <- 0 until leaves.length) {

      val depth = oldToNewDepth(leaves(i).depth)

      /* clear bits */
      for (i <- 0 until bits.length) { 
        bits(i) = 0
      }
      
      /* Keeps track of where we are in new Byte Array */
      var bit = numBits - (depth + 1)
      var byte = bit / 8 
      var shift = 7 - (bit % 8)

      /* Root bit = 1 */
      bits(byte) = bits(byte).|(1 << shift).toByte

      var keep = true
      var j = 0

      while (j < leaves(i).depth) {

        val isRight = leaves(i).lab.testBit(leaves(i).depth - j - 1)

        /* Split is happening in non-conditioned axes, will be represented in new label */
        if (splitToBitNeeded(splitOrder(j)) == 1) {
          bit += 1
          byte = bit / 8
          shift = 7 - (bit % 8)
          if (isRight) {
            bits(byte) = bits(byte).|(1 << shift).toByte 
          }

        /* Keep track of label's box in the slice axes */
        } else {
          val mid = (low(splitOrder(j)) + high(splitOrder(j))) / 2.0
          if (isRight) {
            low(splitOrder(j)) = mid 
          
            /* if slicePoint is no longer in box, the leaf will not be represented in the conditional density */
            if (slicePointExtended(splitOrder(j)) < mid) {
              j = leaves(i).depth
              keep = false
            }
          } else {
            high(splitOrder(j)) = mid 

            /* if slicePoint is no longer in box, the leaf will not be represented in the conditional density */
            if (slicePointExtended(splitOrder(j)) > mid) {
              j = leaves(i).depth
              keep = false
            }
          }
        }

        j += 1
      }

      /* IF (keep) DO update buffers using newLabel, splitBox, update buffer index */
      if (keep) {
        sliceLeavesBuf(bufIndex) = NodeLabel(new BigInt(new BigInteger(bits)))
        val volume = slicedTree.volumeAt(sliceLeavesBuf(bufIndex))
        sliceValuesBuf(bufIndex) = (densMap(leaves(i))._1, volume)
        bufIndex += 1
      }

      /* Reset splitBox */
      densHist.tree.rootCell.low.copyToArray(low)
      densHist.tree.rootCell.high.copyToArray(high)
    }

    if (bufIndex > 0) {
      val slicedNodeMap = (sliceLeavesBuf.take(bufIndex) zip sliceValuesBuf.take(bufIndex)).toMap
      DensityHistogram(slicedTree, fromNodeLabelMap(slicedNodeMap))
    } else {
      null
    }
  }

  def toCollatedHistogram[K](hist: Histogram, key: K): CollatedHistogram[K] = {
    toCollatedHistogram(toDensityHistogram(hist), key)
  }

  def toCollatedHistogram[K](hist: DensityHistogram, key: K): CollatedHistogram[K] = {
    val mappedDensities = hist.densityMap.vals.map(v => Map(key -> v))
    CollatedHistogram(hist.tree, hist.densityMap.copy(vals = mappedDensities))
  }
}

case class CollatedHistogram[K](tree: SpatialTree, densities: LeafMap[Map[K, (Double, Volume)]]) {
  private type MapType = Map[K, (Double, Volume)]

  val keySet = densities.vals.head.keySet

  private def collatorOp(allKeys: Set[K])(kv1: MapType, kv2: MapType): MapType = {
    if (kv1.isEmpty)
      kv2 ++ ((allKeys -- kv2.keySet).map(key => key -> (0.0, 0.0)))
    else if (kv2.isEmpty)
      kv1 ++ ((allKeys -- kv1.keySet).map(key => key -> (0.0, 0.0)))
    else 
      kv1 ++ kv2
  }

  def collate(hist: Histogram, key: K): CollatedHistogram[K] = {
    collate(toCollatedHistogram(hist, key))
  }

  def collateNested(hist: Histogram, key: K): CollatedHistogram[K] = {
    collate(toCollatedHistogram(hist, key), true)
  }

  def collate(hist: DensityHistogram, key: K): CollatedHistogram[K] = {
    collate(toCollatedHistogram(hist, key))
  }

  def collateNested(hist: DensityHistogram, key: K): CollatedHistogram[K] = {
    collate(toCollatedHistogram(hist, key), true)
  }

  def collateNested(hist: CollatedHistogram[K]): CollatedHistogram[K] = {
    collate(hist, true)
  }

  def collate(hist: CollatedHistogram[K], nested: Boolean = false): CollatedHistogram[K] = {
    if (keySet.intersect(hist.keySet).nonEmpty)
      throw new IllegalArgumentException(s"keySets are not allowed to intersect. The common keys are ${keySet intersect hist.keySet}.")

    if (tree != hist.tree)
      throw new IllegalArgumentException("Collated histograms must have the same root box.")

    val allKeys = keySet union hist.keySet
    val base: MapType = Map.empty

    val collatedDensityMap = mrpOperate(densities, hist.densities, collatorOp(allKeys), base, nested)
    CollatedHistogram(
      tree, 
      collatedDensityMap.copy(
        vals = collatedDensityMap.toIterable.toVector.map{ case (node, densMap) =>
          val correctVol = tree.volumeAt(node)
          densMap.mapValues{ case (dens, vol) => (dens, correctVol)
        }  
      })
    )
  }
}
