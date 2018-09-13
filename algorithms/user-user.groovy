import org.lenskit.bias.BiasModel
import org.lenskit.bias.ItemBiasModel
import org.lenskit.knn.user.*
import org.lenskit.knn.NeighborhoodSize
import org.lenskit.transform.normalize.*
import org.grouplens.lenskit.transform.threshold.*

bind ItemScorer to UserUserItemScorer
within (UserVectorNormalizer) {
    bind VectorNormalizer to MeanCenteringVectorNormalizer
}
set NeighborhoodSize to 30
bind (UserSimilarityThreshold, Threshold) to RealThreshold
set ThresholdValue to 1.0e-6