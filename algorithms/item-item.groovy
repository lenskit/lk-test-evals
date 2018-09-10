import org.lenskit.bias.BiasModel
import org.lenskit.bias.ItemBiasModel
import org.lenskit.knn.item.*
import org.lenskit.knn.NeighborhoodSize
import org.lenskit.transform.normalize.BiasUserVectorNormalizer
import org.lenskit.transform.normalize.UserVectorNormalizer
import org.grouplens.lenskit.transform.threshold.*

bind ItemScorer to ItemItemScorer
bind UserVectorNormalizer to BiasUserVectorNormalizer
within (UserVectorNormalizer) {
    bind BiasModel to ItemBiasModel
}
set NeighborhoodSize to 20
set ThresholdValue to 1.0e-6