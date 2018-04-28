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

set IterationCount to 125
set BatchSize to 10000

for (f in [5,10,15,20,25,30,35,40,45,50,60,70,80,90,100]) {
    algorithm("BPR") {
        attributes["Features"] = f
        set FeatureCount to f
    }
}