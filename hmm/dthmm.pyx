#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

"""
Code is based on the following excellent paper from lawrence r. rabbiner
https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf
"""

from .utils cimport logsum_pair
cimport numpy as np
from joblib import Parallel
from joblib import delayed
import numpy as np
import time

DEF NEGINF = float("-inf")
DEF INF = float("inf")

cdef double[:,::1] forward_logprob(const long[::1] observation_sequence,
                                   const double[:,::1] loga,
                                   const double[:,::1] logb,
                                   const double[::1] logpi):
    """
    
    :param observation_sequence: observation sequence
    :param loga: log of transition matrix
    :param logb: log of observation matrix 
    :param logpi: log of initial state probabilities 
    :return: return the logarithm of the forward variable (alpha) 
    """
    T = observation_sequence.shape[0]
    N = loga.shape[0]
    cdef double[:,::1] log_alpha = np.empty((T, N), dtype=np.float64)
    _forward_logprob(observation_sequence, loga, logb, logpi, log_alpha)
    return log_alpha

cdef void _forward_logprob(const long[::1] observation_sequence,
                           const double[:,::1] loga,
                           const double[:,::1] logb,
                           const double[::1] logpi,
                           double[:,::1] log_alpha) nogil:
    cdef int i, j, T, N, t
    cdef double temp
    T = observation_sequence.shape[0]
    N = loga.shape[0]


    for j in range(N):
        log_alpha[0, j] = logpi[j] + logb[j, observation_sequence[0]]

    for t in range(1, T):
        for j in range(N):
            tmp = NEGINF  # log probability of transition from any hidden at t-1 to state to j at t
            for i in range(N):
                tmp = logsum_pair(tmp, log_alpha[t - 1, i] + loga[i, j])
            log_alpha[t, j] = tmp + logb[j, observation_sequence[t]]


cdef double _estimate_observation_logprob(const long[::1] observation_sequence,
                                          const double[:,::1] loga,
                                          const double[:,::1] logb,
                                          const double[::1] logpi):
    """
    estimate the log probability of an observation given model parameters
    
    :param observation_sequence: 
    :param loga: log of transition matrix
    :param logb: log of observation matrix 
    :param logpi: log of initial matrix 
    :return: log probability 
    """

    cdef double[:,::1] log_alpha = forward_logprob(observation_sequence,
                                                   loga, logb, logpi)
    cdef int i, T, N
    T = observation_sequence.shape[0]
    N = loga.shape[0]

    res = NEGINF
    for i in range(N):
        res = logsum_pair(res, log_alpha[T-1, i])

    return res



cdef double[:,::1] backward_logprob(const long[::1] observation_sequence,
                                    const double[:,::1] loga,
                                    const double[:,::1] logb,
                                    const double[::1] logpi):
    """
    
    :param observation_sequence: 
    :param loga: log of transition matrix
    :param logb: log of observation matrix 
    :param logpi: log of initial state probabilities 
    :return: 
    """
    cdef int T = observation_sequence.shape[0]
    cdef int N = loga.shape[0]
    cdef double[:,::1] log_beta = np.empty((T, N), dtype=np.float64)

    _backward_logprob(observation_sequence, loga, logb, logpi, log_beta)

    return log_beta


cdef void _backward_logprob(const long[::1] observation_sequence,
                            const double[:,::1] loga,
                            const double[:,::1] logb,
                            const double[::1] logpi,
                            double[:,::1] log_beta):

    cdef int T = observation_sequence.shape[0]
    cdef int N = loga.shape[0]
    cdef int i,t,j

    #log_beta[-1,:] = 0  #log(1) = 0
    for i in range(N):
        log_beta[T - 1, i] = 0

    for t in range(T - 2, -1, -1):
        for i in range(N):
            tmp = NEGINF
            for j in range(N):
                tmp = logsum_pair(tmp,
                                  log_beta[t + 1, j] +
                                  loga[i, j] + logb[j, observation_sequence[t + 1]])
            log_beta[t, i] = tmp


cpdef double[:,::1] state_logprob(const double[:,::1] log_alpha,
                                  const double[:,::1] log_beta):
    """
    log probability of being at state s at time t
    :param alpha: forward variable alpha of a given observation sequence
    :param beta: backward variable beta of the same observation sequence 
    :return: 
    """
    cdef int T, N

    T = log_alpha.shape[0]
    N = log_alpha.shape[1]
    cdef double[:,::1] log_gamma = np.empty((T, N), dtype=np.float64)

    _state_logprob(log_alpha, log_beta, log_gamma)

    return log_gamma

cdef void _state_logprob(const double[:,::1] log_alpha,
                         const double[:,::1] log_beta,
                         double[:,::1] log_gamma):
    cdef int T, N, t, i, j

    T = log_alpha.shape[0]
    N = log_alpha.shape[1]

    for t in range(T):
        for i in range(N):
            log_gamma[t, i] = log_alpha[t, i] + log_beta[t, i]
            tmp = NEGINF
            for j in range(N):
                tmp = logsum_pair(tmp, log_alpha[t, j] + log_beta[t, j])
            log_gamma[t, i] -= tmp


cpdef double[:,:,::1] double_state_prob(const long[::1] observation_sequence,
                                        const double[:,::1] log_alpha,
                                        const double[:,::1] log_beta,
                                        const double[:,::1] loga,
                                        const double[:,::1] logb):
    """
    log probability from transition to two states at time t
    :param observation_sequence: 
    :param log_alpha: 
    :param log_beta: 
    :param loga: log of transition matrix
    :param logb: log of observation matrix 
    :return: 
    """

    cdef int N, T

    N = loga.shape[0]
    T = observation_sequence.shape[0]

    cdef double[:,:,::1] ksi = np.empty((T - 1, N, N), dtype=np.float64)
    _double_state_prob(observation_sequence, log_alpha, log_beta, loga, logb, ksi)
    return ksi

cdef void _double_state_prob(const long[::1] observation_sequence,
                             const double[:,::1] log_alpha,
                             const double[:,::1] log_beta,
                             const double[:,::1] loga,
                             const double[:,::1] logb,
                             double[:,:,::1] ksi):
    """
    
    :param observation_sequence: 
    :param log_alpha: 
    :param log_beta: 
    :param loga: log of transition matrix
    :param logb: log of observation matrix 
    :param ksi: 
    :return: 
    """

    cdef int N, T, t, i, j

    N = loga.shape[0]
    T = observation_sequence.shape[0]

    cdef double tmp

    for t in range(T - 1):
        tmp = NEGINF
        for i in range(N):
            for j in range(N):
                tmp = logsum_pair(tmp, log_alpha[t, i] + loga[i, j] + logb[j, observation_sequence[t + 1]] + log_beta[t + 1, j])
                ksi[t, i, j] = log_alpha[t, i] + loga[i, j] + logb[j, observation_sequence[t + 1]] + log_beta[t + 1, j]
        for i in range(N):
            for j in range(N):
                ksi[t, i, j] -= tmp


cdef class BaumWelchBatch:

    cdef public double[::1] log_pi_sum
    #expected number of transitions from State Si
    cdef public double[::1] log_gamma_sum
    #expected number of times in sate j
    cdef public double[::1] log_gamma_full_sum
    #expected number of transitions from State Si to Sate Sj
    cdef public double[:,::1] log_ksi_sum
    #expected number of times in States j and observing symbol v_k
    cdef public double[:,::1] log_obs_sum
    #
    cdef public int stat_size
    cdef double[:,::1] loga
    cdef double[:,::1] logb
    cdef double[::1] logpi

    def __init__(self,
                 double[:,::1] loga,
                 double[:,::1] logb,
                 double[::1] logpi
                 ):
        cdef int N = loga.shape[0]
        cdef int M = logb.shape[1]
        self.loga = loga
        self.logb = logb
        self.logpi = logpi
        self.log_pi_sum = np.full(N, NEGINF, dtype=np.float64)
        self.log_ksi_sum = np.full((N,N), NEGINF, dtype=np.float64)
        self.log_gamma_sum = np.empty(N, dtype=np.float64)
        self.log_obs_sum = np.full((N, M), NEGINF, dtype=np.float64)
        self.log_gamma_full_sum = np.full(N, NEGINF, dtype=np.float64)
        self.stat_size = 0

    def __getstate__(self):
        state= {}
        state['loga'] = np.asarray(self.loga)
        state['logb'] = np.asarray(self.logb)
        state['logpi'] = np.asarray(self.logpi)
        state['log_pi_sum'] = np.asarray(self.log_pi_sum)
        state['log_gamma_sum'] = np.asarray(self.log_gamma_sum)
        state['log_gamma_full_sum'] = np.asarray(self.log_gamma_full_sum)
        state['log_ksi_sum'] = np.asarray(self.log_ksi_sum)
        state['log_obs_sum'] = np.asarray(self.log_obs_sum)
        state['batch_size'] = self.stat_size
        return state

    def __setstate__(self, state):
        self.logpi = state['logpi']
        self.loga = state['loga']
        self.logb = state['logb']
        self.log_pi_sum = state['log_pi_sum']
        self.log_gamma_sum = state['log_gamma_sum']
        self.log_gamma_full_sum = state['log_gamma_full_sum']
        self.log_ksi_sum = state['log_ksi_sum']
        self.log_obs_sum = state['log_obs_sum']
        self.stat_size = state['batch_size']

    def __reduce__(self):
        state=self.__getstate__()
        return self.__class__, (state['loga'], state['logb'], state['logpi'],), state

    cpdef void fit_sequence(self, const long[::1] observation_sequence):
        """
        fit's an obseravtion sequence and set's log of the following statistics pi_sum, 
        gamma_sum, gamma_full_sum, ksi_sum, obs_sum
        mainly used for parallel processing 
        :param observation_sequence: 
        :return: 
        """
        #_forward_logprob(data[row_index], self.loga, self.logb, self.logpi, log_alpha)
        cdef double[:,::1] log_alpha = forward_logprob(observation_sequence, self.loga, self.logb, self.logpi)
        cdef double[:, ::1] log_beta = backward_logprob(observation_sequence, self.loga, self.logb, self.logpi)
        #_backward_logprob(data[row_index], self.loga, self.logb, self.logpi, log_beta)

        cdef double[:, ::1] log_gamma = state_logprob(log_alpha, log_beta)
        #_state_logprob(log_alpha, log_beta, log_gamma)
        cdef double[:,:,::1] log_ksi = double_state_prob(observation_sequence,log_alpha, log_beta, self.loga, self.logb)
        #_double_state_prob(data[row_index], log_alpha, log_beta, self.loga, self.logb, log_ksi)

        cdef int T = observation_sequence.shape[0]
        cdef int N = self.loga.shape[0]
        cdef int M = self.logb.shape[1]


        for i in range(N):
            self.log_pi_sum[i] = logsum_pair(self.log_pi_sum[i], log_gamma[0,i])

        #expected number of transition from State Si to Sate Sj
        for i in range(N):
            for j in range(N):
                tmp = NEGINF
                for t in range(T-1):
                    tmp = logsum_pair(tmp, log_ksi[t, i,j])
                self.log_ksi_sum[i, j] = logsum_pair(tmp, self.log_ksi_sum[i, j])

        #expected number of transition from State Si
        for i in range(N):
            tmp = NEGINF
            for t in range(T-1):
                tmp = logsum_pair(tmp, log_gamma[t, i])
            self.log_gamma_sum[i] = tmp

        #expected number of times in States j and observing symbol v_k
        for t in range(T):
            for j in range(N):
                #data[row_index][t] is k in paper
                self.log_obs_sum[j, observation_sequence[t]] = logsum_pair(self.log_obs_sum[j, observation_sequence[t]],
                                                                           log_gamma[t, j])

        #expected number of times in state j
        for i in range(N):
            tmp = NEGINF
            for t in range(T):
                tmp = logsum_pair(tmp, log_gamma[t, i])
            self.log_gamma_full_sum[i] = tmp


        self.stat_size+= 1



cdef BaumWelchBatch combine_run_pair(BaumWelchBatch brun1,
                                     BaumWelchBatch brun2):
    """
    combines two baum welch run's (brun1 and brun2)
    into a new BaumWechBatch instance
    
    :param brun1: BaumWelchBatch instance
    :param brun2: BaumWelchBatch instance 
    :return: BaumWelchBatch instance that combines brun1 and brun2
    """

    cdef BaumWelchBatch out = BaumWelchBatch(brun1.loga, brun1.logb, brun1.logpi)

    cdef int N = brun1.log_pi_sum.shape[0]
    cdef int M = brun1.log_obs_sum.shape[1]

    for i in range(N):
        out.log_pi_sum[i] = logsum_pair(brun1.log_pi_sum[i],
                                        brun2.log_pi_sum[i])
        out.log_gamma_sum[i] = logsum_pair(brun1.log_gamma_sum[i],
                                           brun2.log_gamma_sum[i])
        out.log_gamma_full_sum[i] = logsum_pair(brun1.log_gamma_full_sum[i],
                                                brun2.log_gamma_full_sum[i])

    for i in range(N):
        for j in range(M):
            out.log_obs_sum[i][j] = logsum_pair(brun1.log_obs_sum[i][j],
                                                brun2.log_obs_sum[i][j])
        for j in range(N):
            out.log_ksi_sum[i][j] = logsum_pair(brun1.log_ksi_sum[i][j],
                                                brun2.log_ksi_sum[i][j])

    out.stat_size = brun1.stat_size + brun2.stat_size

    return out


cdef HiddenMarkovModel combine_batches(BaumWelchBatch[:] baum_welch_batches):

    cdef int i, j
    cdef BaumWelchBatch res = baum_welch_batches[0]

    for i in range(baum_welch_batches.shape[0]):
        res = combine_run_pair(res, baum_welch_batches[i])

    cdef int N = res.log_pi_sum.shape[0]
    cdef int M = res.log_obs_sum.shape[1]

    cdef double[::1] logpi = np.empty(N, dtype=np.float64)
    cdef double[:,::1] loga = np.empty((N, N), dtype=np.float64)
    cdef double[:,::1] logb = np.empty((N, M), dtype=np.float64)

    for i in range(N):
        logpi[i] = res.log_pi_sum[i] - np.log(res.stat_size)
        for j in range(N):
            loga[i,j] = res.log_ksi_sum[i,j] - res.log_gamma_sum[i]
        for j in range(M):
            logb[i,j] = res.log_obs_sum[i,j] - res.log_gamma_full_sum[i]

    return HiddenMarkovModel(np.exp(np.array(loga)),
                             np.exp(np.array(logb)),
                             np.exp(np.array(logpi)))




cdef class HiddenMarkovModel:
    cdef double[:,::1] _loga
    cdef double[:,::1] _logb
    cdef double[::1] _logpi

    @property
    def start_probabilities(self):
        return np.exp(self._logpi)

    @property
    def transition_probabilities(self):
        return np.exp(self._loga)

    @property
    def observation_probabilities(self):
        return np.exp(self._logb)

    @property
    def loga(self):
        return self._loga

    @property
    def logb(self):
        return self._logb

    @property
    def logpi(self):
        return self._logpi

    def __init__(self, transition_probabilities, observation_probabilities,
                 start_probabilities, states=None, symbols=None):
        self._loga = np.log(transition_probabilities)
        self._logb = np.log(observation_probabilities)
        self._logpi = np.log(start_probabilities)

    def __getstate__(self):
        state = {}
        state['loga'] = np.asarray(self._loga)
        state['logb'] = np.asarray(self._logb)
        state['logpi'] = np.asarray(self._logpi)
        return state

    def __setstate__(self, state):
        self._logpi = state['logpi']
        self._loga = state['loga']
        self._logb = state['logb']

    def __reduce__(self):
        state = self.__getstate__()
        return self.__class__, (np.exp(state['loga']), np.exp(state['logb']), np.exp(state['logpi']),), state

    def forward(self, observation_sequence):
        return np.exp(forward_logprob(observation_sequence, self._loga, self._logb, self._logpi))

    def backward(self, observation_sequence):
        return np.exp(backward_logprob(observation_sequence, self._loga, self._logb, self._logpi))

    cpdef double observation_log_probability(self, long[::1] observation_sequence):
        return _estimate_observation_logprob(observation_sequence, self._loga, self._logb, self._logpi)

    @classmethod
    def from_batches(cls, batches):
        """
        return's a HiddenMarkovModel that is the combination of multiple BaumWelchBatch runs that
        are usually run in parallel

        :param batches: list of BaumWelchBatch instances
        :return: HiddenMarkovModel instance
        """
        return combine_batches(np.array(batches))

    def sample(self, n=None, length=None, random_state=None):
        """
        returns hidden states and sequences generated from the model

        :param n:  number of samples, default is 1
        :param length: length of each sequence
        :param random_state: random state used for generating sequences
        :return: hidden states, sequences
        """
        if random_state is None:
            random_state = np.random.get_state()[1][0]
        np.random.seed(random_state)

        if length is None:
            length = np.random.choice(range(5, 30), 1)

        if n is None:
            return self._generate_sequence(length, random_state)
        else:
            return [self._generate_sequence(length, random_state) for _ in range(n)]

    cdef tuple _generate_sequence(self, int length, long random_state):

        #np.random.seed(random_state)


        cdef np.ndarray out_states = np.empty(length, dtype=np.int)
        cdef np.ndarray out_sequence = np.empty(length, dtype=np.int)

        cdef int N = self._loga.shape[0]
        cdef int M = self._logb.shape[1]

        cdef np.ndarray A =  np.exp(self._loga)
        cdef np.ndarray B =  np.exp(self._logb)
        cdef np.ndarray pi =  np.exp(self._logpi)

        cdef int current_state = np.random.choice(N, size=1, p=pi)

        for i in range(length):
            out_states[i] = current_state
            out_sequence[i] = np.random.choice(M, size=1, p=B[current_state,:].flatten())
            current_state = np.random.choice(N, size=1, p=A[current_state,:].flatten())

        return out_states, out_sequence

    def fit(self, data, jobs=-1, batch_size=1000, stop_threshold=1E-9, min_iterations=0, max_iterations=1000):
        """
        estimate model parameter using the baum-welch algorithm
        :param data:
        :param jobs:
        :param batch_size:
        :param stop_threshold:
        :param min_iterations:
        :param max_iterations:
        :return:
        """
        def generate_batches(data, batch_size):
            start, end = 0, batch_size
            while start < len(data):
                yield data[start:end]
                start += batch_size
                end += batch_size

        def fit_worker(data):
            batch = BaumWelchBatch(self._loga, self._logb, self._logpi)
            for d in data:
                batch.fit_sequence(d)
            return batch


        with Parallel(n_jobs=-1, backend='threading') as parallel:
            f2 = delayed(lambda x: sum(map(self.observation_log_probability, x)))
            log_prob_sum = sum(parallel(f2(batch) for batch in generate_batches(data, batch_size)))

        iteration = 0
        improvement = INF

        while improvement > stop_threshold or iteration < min_iterations + 1:
            print(f'iteration {iteration +1}')
            s = time.time()

            with Parallel(n_jobs=-1, backend='threading') as parallel:
                f = delayed(fit_worker)
                baum_welch_runs = parallel(f(batch) for batch in generate_batches(data, batch_size))
                new_model = HiddenMarkovModel.from_batches(baum_welch_runs)
                f2 = delayed(lambda x: sum(map(new_model.observation_log_probability, x)))
                new_logprob_sum = sum(parallel(f2(batch) for batch in generate_batches(data, batch_size)))
                improvement = new_logprob_sum - log_prob_sum
                print(f'improvement = {improvement:.18f}')
                log_prob_sum = new_logprob_sum
                if iteration >= max_iterations:
                    break
                iteration += 1

            e = time.time()
            print(f'took {e-s}s')
