import org.grouplens.lenskit.iterative.IterationCount
import org.lenskit.api.ItemScorer
import org.lenskit.bias.BiasModel
import org.lenskit.bias.ZeroBiasModel
import org.lenskit.mf.BiasedMFItemScorer
import org.lenskit.mf.MFModel
import org.lenskit.mf.bpr.BPRMFModelProvider
import org.lenskit.mf.bpr.BatchSize
import org.lenskit.mf.bpr.RandomRatingPairGenerator
import org.lenskit.mf.bpr.TrainingPairGenerator
import org.lenskit.mf.funksvd.FeatureCount

bind ItemScorer to BiasedMFItemScorer
bind BiasModel to ZeroBiasModel
bind MFModel toProvider BPRMFModelProvider
bind TrainingPairGenerator to RandomRatingPairGenerator

set FeatureCount to 25
set IterationCount to 125
set BatchSize to 10000