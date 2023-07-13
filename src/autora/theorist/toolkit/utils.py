import random

##################
#       Functions
##################


# function to build Tree from Expression
# def build_tree():
#     ...

# def parse_expr_str(expr_str):
#     parsed_expr = []


###
# decision_sample
###
# takes: takes a list of lists of floats corresponding to weights for choices
# gives: list of integers, with each integer corresponding to the index chosen for each list of floats given
def decision_sample(search_space):
    sample = []
    for options in search_space:
        sample.append(random.choices(range(len(options)), weights=options, k=1))
    return sample


###
# (hamiltonian) mcmc sample
###
# takes: model, search space with markov chain weights, number of hamiltonian steps (probability for bernoulli dist)
# gives: new model
# def mcmc_sample():
#    ...



