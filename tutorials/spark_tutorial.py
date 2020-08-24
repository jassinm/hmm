from hmm.dthmm import BaumWelchBatch
from pyspark.sql import SparkSession
from hmm import HiddenMarkovModel
import time
import findspark
import pickle
import logging

logger = logging.getLogger(__name__)

findspark.init()


def fit(sc, data, model, stop_threshold=1E-9,
        min_iterations=0, max_iterations=1000):
    """

    fits HIddenMarkovModel (model) using baum welch to a set of data (data) in parrallel using SparkContext (sc)
    until either a threshold (stop_threshold) is met or at least min_iterations are run.
    max_iterations is used to stop the fitting process.


    :param sc: sparkContext
    :param data: data to be fitted
    :param model: HiddenMarkovModel instance
    :param stop_threshold:
    :param min_iterations: minimum number of iterations
    :param max_iterations: maximum number of iterations

    :return: HiddenMarkovModel instances fitted to the data. Use model to extract estimated model parameters
    """

    def fit_worker(batch):
        bwelch = BaumWelchBatch(model.loga, model.logb, model.logpi)
        for d in batch:
            bwelch.fit_sequence(d)
        return [bwelch]

    p_data = sc.parallelize(data)
    log_prob_sum = p_data.map(model.observation_log_probability).reduce(lambda x, y: x + y)
    iteration = 0
    improvement = float('inf')
    new_model = None

    while improvement > stop_threshold or iteration < min_iterations + 1:
        s = time.time()
        batches = p_data.mapPartitions(fit_worker).collect()
        logger.info(f'got baches of size {len(batches)}')
        new_model = HiddenMarkovModel.from_batches(batches)
        new_log_prob_sum = p_data.map(new_model.observation_log_probability).reduce(lambda x, y: x + y)
        improvement = new_log_prob_sum - log_prob_sum
        e = time.time()
        logger.info(f'took {e - s}')
        logger.info(f'improvement = {improvement:.5f}')
        log_prob_sum = new_log_prob_sum
        if iteration >= max_iterations:
            break
        iteration += 1

    return new_model


def main():
    spark = (SparkSession.builder
             .master("local[*]")
             .config("spark.executor.memory", "30g")
             .config("spark.driver.memory", "30g")
             .config("spark.driver.maxResultSize", "30g")
             .config("spark.memory.offHeap.enabled", True)
             .config("spark.memory.offHeap.size", "16g")
             .appName("sampleCodeForReference")
             .getOrCreate())

    sc = spark.sparkContext

    test_data = pickle.load(open('/Users/locojay/PycharmProjects/dthmm/tutorials/testdata2.p', 'rb'))

    A = [[0.17299125, 0.08781199, 0.24904337, 0.49015339],
         [0.65466035, 0.0058856, 0.24847472, 0.09097933],
         [0.43406668, 0.09507003, 0.24143807, 0.22942522],
         [0.00310297, 0.41726041, 0.27046179, 0.30917482]]

    B = [[0.0248371, 0.00647766, 0.02919312, 0.02010902, 0.01741969, 0.03026002,
          0.01107451, 0.03090185, 0.02000882, 0.02946754, 0.0329583, 0.02810143,
          0.00973118, 0.01286111, 0.03036823, 0.03451904, 0.01301527, 0.03176073,
          0.02069127, 0.0391591, 0.03724013, 0.01681755, 0.02387927, 0.01267418,
          0.01405466, 0.00182615, 0.00099688, 0.02921965, 0.02068266, 0.00459763,
          0.03083269, 0.02294538, 0.00748594, 0.0318249, 0.01643839, 0.03030681,
          0.00853397, 0.02212386, 0.02451805, 0.01147829, 0.01860806, 0.01689099,
          0.01947854, 0.00456117, 0.01985139, 0.02348703, 0.02722838, 0.02259387,
          0.00460825, 0.00130027],
         [0.00118511, 0.0364538, 0.00539255, 0.02931715, 0.00712114, 0.02613686,
          0.02025734, 0.00856556, 0.01788003, 0.02696186, 0.03206167, 0.02082036,
          0.02027708, 0.0363248, 0.01253547, 0.02536659, 0.0303423, 0.00161272,
          0.02162873, 0.0211614, 0.01741675, 0.01470692, 0.032151, 0.03228765,
          0.03237699, 0.0370071, 0.01195834, 0.02739508, 0.01974688, 0.01438907,
          0.00741205, 0.02553209, 0.00501492, 0.02914962, 0.01528311, 0.02546899,
          0.01965691, 0.00166134, 0.0146325, 0.03175253, 0.00425995, 0.02717155,
          0.02544106, 0.03355649, 0.02468158, 0.00874545, 0.01172551, 0.02154314,
          0.00843848, 0.01803447],
         [0.0296181, 0.0348821, 0.02564371, 0.02800763, 0.01551197, 0.02558589,
          0.03501015, 0.01300263, 0.01266429, 0.03546458, 0.00678947, 0.01032237,
          0.03453364, 0.02323215, 0.01534716, 0.03644205, 0.02687086, 0.02292363,
          0.00105033, 0.0289615, 0.02795536, 0.03250376, 0.02837804, 0.01249522,
          0.02217764, 0.02628832, 0.00928285, 0.00739886, 0.03279007, 0.00722151,
          0.00053051, 0.01206393, 0.01819556, 0.00779652, 0.02419107, 0.00798948,
          0.00664281, 0.02770423, 0.0339964, 0.01410592, 0.01401967, 0.03120296,
          0.02565983, 0.01024386, 0.01415742, 0.00839726, 0.01779137, 0.02100865,
          0.02521129, 0.01073536],
         [0.01471172, 0.02670568, 0.01813862, 0.03895738, 0.0074108, 0.00734445,
          0.02980466, 0.0244879, 0.00582519, 0.0089145, 0.00959946, 0.02949902,
          0.01730438, 0.00265082, 0.00898055, 0.00310906, 0.02095744, 0.02549341,
          0.00517031, 0.01065439, 0.03255066, 0.03373455, 0.00429001, 0.0298808,
          0.03904555, 0.00203563, 0.0188991, 0.02278372, 0.02672836, 0.01151306,
          0.01512417, 0.03303694, 0.03390606, 0.02449836, 0.01443768, 0.0127056,
          0.03821532, 0.01233168, 0.00493174, 0.03505321, 0.03774991, 0.03070529,
          0.02777502, 0.00753259, 0.02052302, 0.02192132, 0.00473921, 0.03786516,
          0.03214382, 0.01762273]]

    pi = [0.1785823, 0.20446237, 0.26092583, 0.3560295]
    model = HiddenMarkovModel(A, B, pi)
    fit(sc, test_data, model)


if __name__ == '__main__':
    import sys
    sys.exit(main())
