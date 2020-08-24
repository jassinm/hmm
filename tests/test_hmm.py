from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
from hmm import HiddenMarkovModel
from hmm.dthmm import state_logprob
from hmm.dthmm import double_state_prob
import numpy as np
import pytest


def generate_possible_sequences(set_vals, sequence_length):
    def generate_possible_sequence_h(k):
        if k == 0:
            yield []
        else:
            for v in set_vals:
                for pos in generate_possible_sequence_h(k - 1):
                    yield [v] + pos

    return generate_possible_sequence_h(sequence_length)


def log_prob_sequence_brute_force(observed, logA, logB, logpi):
    from scipy.special import logsumexp
    M_labels = logB.shape[0]
    res = []
    all_possible_state_sequences = generate_possible_sequences(range(M_labels), len(observed))

    for potential_state_sequence in list(all_possible_state_sequences):
        states_observations = zip(potential_state_sequence, observed)
        s0, O0 = next(states_observations)
        logprob = logpi[s0] + logB[s0, O0]
        prevstate = s0
        for state, obs in states_observations:
            logprob += logA[prevstate, state] + logB[state, obs]
            prevstate = state
        res.append(logprob)
    return logsumexp(res)


@pytest.fixture
def simple_hmm():
    A = np.array([[0.9, 0.1],
                  [0.4, 0.6]])
    B = np.array([[0.9, 0.1],
                  [0.2, 0.8]])
    pi = np.array([0.8, 0.2])

    model = HiddenMarkovModel(A, B, pi)
    emissions = np.array([0, 1])
    return (model, emissions)


@pytest.fixture
def short_emission(simple_hmm):
    """Return DtHMM and medium emission sequence"""
    hmm, em = simple_hmm
    em = np.array([0, 1, 1])
    return (hmm, em)


def test_forward(simple_hmm):
    model, emissions = simple_hmm
    O = np.array([[0.72, 0.04],
                  [0.0664, 0.0768]])
    X = model.forward(emissions)
    assert_array_almost_equal(O, X)


def test_backward(simple_hmm):
    model, emissions = simple_hmm
    O = np.array([[0.17, 0.52],
                  [1, 1]])
    X = model.backward(emissions)
    assert_array_almost_equal(O, X)


def test_estimate(simple_hmm):
    model, emissions = simple_hmm
    a = 0.1432
    assert_almost_equal(np.exp(model.observation_log_probability(emissions)), a)
    assert_almost_equal(np.exp(model.observation_log_probability(emissions)),
                        np.exp(log_prob_sequence_brute_force(emissions,
                                                             model.loga,
                                                             model.logb,
                                                             model.logpi)))


def test_state_prob(short_emission):
    model, emissions = short_emission

    alpha = model.forward(emissions)
    beta = model.backward(emissions)
    res = np.exp(state_logprob(np.log(alpha), np.log(beta)))
    gamma_out = np.array([[0.79978135, 0.20021865], [0.22036545, 0.77963455], [0.17663595, 0.82336405]])
    # assert_array_almost_equal(model.state_prob(emissions), gamma_out)
    assert_array_almost_equal(res, gamma_out)


#
def test_double_state_prob(short_emission):
    model, emissions = short_emission
    ksi_out = np.array([[0.11666406, 0.10370139], [0.05997189, 0.71966266]])
    alpha = model.forward(emissions)
    beta = model.backward(emissions)
    loga = np.log(model.transition_probabilities)
    logb = np.log(model.observation_probabilities)
    res = np.exp(double_state_prob(emissions, np.log(alpha), np.log(beta), loga, logb))
    # print(model.double_state_prob(emissions))
    # assert_array_almost_equal(model.double_state_prob(emissions)[1, :], ksi_out)
    assert_array_almost_equal(res[1, :], ksi_out)


def test_fit(simple_hmm):
    model, emissions = simple_hmm
    data = np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]])
    model.fit(data, max_iterations=10)
    print(model.observation_probabilities)
    print(model.transition_probabilities)
    print(model.start_probabilities)


def test_s_pickable(simple_hmm):
    from hmm.dthmm import BaumWelchBatch
    import pickle
    model, em = simple_hmm
    A = model.transition_probabilities
    B = model.observation_probabilities
    pi = model.start_probabilities

    bwelch = BaumWelchBatch(A, B, pi)
    bwelch.fit_sequence(em)
    bwelch_pickled = pickle.dumps(bwelch)
    bwelch_unpickled = pickle.loads(bwelch_pickled)
    assert_array_almost_equal(bwelch.log_pi_sum, bwelch_unpickled.log_pi_sum)

    model_pickled = pickle.dumps(simple_hmm)
    model_unpickled = pickle.loads(bwelch_pickled)

