import org.lenskit.bias.BiasModel
import org.lenskit.bias.ItemBiasModel
import org.lenskit.knn.item.ItemItemScorer
import org.lenskit.knn.NeighborhoodSize
import org.lenskit.transform.normalize.BiasUserVectorNormalizer
import org.lenskit.transform.normalize.UserVectorNormalizer

bind ItemScorer to ItemItemScorer
bind UserVectorNormalizer to BiasUserVectorNormalizer
within (UserVectorNormalizer) {
    bind BiasModel to ItemBiasModel
}
set NeighborhoodSize to 20