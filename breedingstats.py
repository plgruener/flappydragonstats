#!/usr/bin/env python

import math
import json
import itertools
from operator import itemgetter
from pprint import pprint
import pickle

################################################################################

### SHINY CHANCE ###############################################################
# How often (mean) to breed a pair until the shiny-chance reaches the cap of 100(=10%)?
# Each breeding seems to increase the value by a random number between 1-10 (inclusive),
# thus equivalent to: how often to roll a 10-sided die until cumulative sum reaches 100?
# see https://nickarnosti.com/blog/rollto100/

# how often to roll a d-sided die until cumulative sum reaches the target t
def compute_x(d,t):
    M = [[0 for j in range(t-1)] for i in range(t-1)]
    for k in range(d):
        M[0][k]=1
    for i in range(1,t-1): # first row already filled
        for j in range(1,t-1): # first column cannot be filled, is all 0
            if j>=d:
                M[i][j] = sum([ M[i-1][k] for k in range(j-d,j) ])
            else:
                M[i][j] = sum([ M[i-1][k] for k in range(0,j) ])

    return 1+sum([ sum(row)/d**i for i,row in enumerate(M,1) ])

#print(compute_x(6,100)) # =29.047619..., same as in the article
#print(compute_x(10,100)) # =18.727272..., matches the intuition since the mean of 1..10 is 5.5 and 100/5.5 = 18.181818

# the total mean shiny-probability is thus
compute_p_shiny = lambda x: 0.1*( (x/2+10)/(x+10) ) # where x is the mean number of steps until 100

p_shiny = compute_p_shiny(compute_x(10,100)) # = 0.06740506 # 1/p=14.836795

### GENERATE PAIRINGS ##########################################################

# number N of all possible pairings of n (n even) dragons [(d1,d2),(d3,d4),…(dn-1,dn)],
# without order of or in the pairs,r
# N = n!/( 2!^(n/2) * (n/2)! ) = (n-1)*(n-2)*…*5*3*1
def number_pairings(n):
    return math.prod(range(1,n,2))

# generate all the possible permutations of pairings of n dragons without duplicates
#
# First, naive approach with a recursive function was too memory-intensive for n=20
# Instead we can enumerate each combination by counting from 0 to N-1
# in the mixed radix numeral system with bases [n-1,n-3,…,7,5,3],
# because N=(n-1)*(n-3)*…*7*5*3.
# https://en.wikipedia.org/wiki/Mixed_radix

def dec2mixradix(n,bases):
    out=[]
    for base in reversed(bases):
        out=[n%base]+out
        n=n//base
    return out

# input: number k in mixed radix system to bases [n-1,…,7,5,3]
def gen_kth_pairing(l,indices):
    res=[]
    lr=l[:] #copy
    for idx in indices:
        idx=idx+1 # +1 because the first element,l[0], is already set
        res.append((lr[0],lr[idx]))
        lr = lr[1:idx]+lr[idx+1:]
    res.append((lr[0],lr[1])) # unroll the last (trivial) recursion
    return res

def gen_pairings(l):
    # count from 0 to N, directly in the mixed radix system,
    # use those digits as indices to generate the k-th pairing
    return ( gen_kth_pairing(l,i) for i in itertools.product(*[ list(range(m)) for m in range(len(l)-1,2,-2)]) )

### HELPER FUNCTIONS ###########################################################
# [n1,n2,n3,n4,…] -> [(n1,n2),(n3,n4),…]
def list2pairs(l:list)->list:
    assert len(l)%2==0
    return [ (l[i],l[i+1]) for i in range(0,len(l),2) ]
# [(n1,n2),(n3,n4),…] -> [n1,n2,n3,n4,…]
def pairs2list(l:list)->list:
    return list(itertools.chain.from_iterable(l))
# sort list of tuples by other index
def sorted_tuplelist(l,i,reverse=True):
    return sorted(l,key=itemgetter(i),reverse=reverse)

################################################################################

### DRAGON DATA ################################################################
with open('dragons.json') as f:
    dragon_dict = json.load(f)
dragon_names:list = sorted([dragon['name'] for dragon in dragon_dict])

def get_dragon(name: str) -> dict:
    return next(dragon for dragon in dragon_dict if dragon['name']==name)
def dragon_name(dragon:dict)->str:
    return dragon['name']
def dragon_rarity(dragon:str)->str:
    return get_dragon(dragon)['rarity']
def dragon_eggs(dragon:str)->list:
    return get_dragon(dragon)['breeding_eggs']

# value = number of crowns for duplicated dragons
# RARITY    TIME   VALUE
# Common     1       40
# Uncommon   2       80
# Rare       4      160
# Epic       8      320
# Legendary 12      640
# Mythic    24     1280
rarities = ['Common','Uncommon','Rare','Epic','Legendary','Mythic']
def dragon_value(dragon: str) -> int:
    return 40*2**rarities.index(dragon_rarity(dragon))

### EGG DATA ###################################################################
# generate the list of eggs one can get from breeding and the dragons they hatch into
eggs=dict()
for dragon in dragon_names:
    for egg in dragon_eggs(dragon):
        if not egg in eggs:
            eggs[egg]=set()
        eggs[egg].add(dragon)

# set of possible dragons to hatch from an egg
def egg_pool(egg:str)->set:
    return eggs[egg]
def egg_value(egg:str)->float:
    return sum([dragon_value(dragon) for dragon in egg_pool(egg)])/len(egg_pool(egg))

### BREEDING ###################################################################

def breeding_value(d1:str,d2:str,p=p_shiny):
    assert 0<=p<=0.1 # currently cannot be >10%
    eggs = set(dragon_eggs(d1)+dragon_eggs(d2)) # union set = no duplicates
    # p: chance of breeding a shiny which itself has a 50/50 chance between each parent
    return p*(0.5*dragon_value(d1)+0.5*dragon_value(d2)) + (1-p)*(sum([egg_value(e) for e in eggs])/len(eggs))

# [name,name,…] or [(name,name),(…),…]
def breeding_sum(list_or_pairing,p=p_shiny):
    if isinstance(list_or_pairing[0],str):
        breeding_sum_list(list_or_pairing,p)
    if isinstance(list_or_pairing[0],tuple):
        breeding_sum_pairing(list_or_pairing,p)
def breeding_sum_pairing(pairs,p=p_shiny):
    return sum([breeding_value(d1,d2,p) for (d1,d2) in pairs])
def breeding_sum_list(list,p=p_shiny):
    return breeding_sum_pairing(list2pairs(l),p)

### LOOKUP TABLE ###############################################################

# pre-compute a lookup table of breeding_values for each possible dragon pair
# (all permutations with duplication for ease of use)
def generate_lookup_table(p=p_shiny)->dict:
    lookup = dict()
    for d1 in dragon_names:
        for d2 in dragon_names:
            lookup[(d1,d2)] = breeding_value(d1,d2,p)
    return lookup

# lookup generation takes a minute, so we might want to save the table to file
try:
    with open('breeding_lookup.pickle','rb') as f:
        lookup = pickle.load(f)
except FileNotFoundError:
    print('Cannot find lookup table file, re-generating.')
    lookup=generate_lookup_table() # still pretty fast
    with open('breeding_lookup.pickle','wb') as f:
        pickle.dump(lookup,f)

# now we can use the lookup table to redefine the breeding_sum function
# to avoid repeated computation in our search function below
# NOTE: the lookup table was only made for one p value
def breeding_sum_lookup(name_pairs,lookup=lookup):
    return sum([lookup[(d1,d2)] for (d1,d2) in name_pairs])

### SEARCH PAIRINGS ############################################################
# Given a list of n(=20) dragons, iterate over all possible pairings and
# track the best total breeding value.
#
# NOTE: Two (or more) dragons with the same egg groups are swappable, meaning
#       (X,A)+(X',B)==(X',A)+(X,B).
# This leads to a lot of duplication to search through, and a lot of pairings
# with identical value.
#TODO adapt gen_pairings algorithm to take a MultiSet as input and output only
# the unique pairings. Seems to be a hard problem?
#

#TODO unify lookup- and p- parameter usage
def search_all(dragonlist,lookup=lookup,p=p_shiny):
    print(f'{len(dragonlist)=} {dragonlist=}')
    num=number_pairings(len(dragonlist)) # == 654_729_075 = 6.5E8 for n=20
    bestval=0
    bestpairing=[]

    for i,pk in enumerate(gen_pairings(dragonlist)):
        if i%1_000_000==0: #adjust if necessary
            print(f'progress: {i/num:.4f}') # very simple progress indicator
        r = breeding_sum_lookup(pk,lookup)
        if r > bestval:
            bestval=r
            bestpairing=pk
            print(f'==> NEW BEST: {bestval}')
            print_result(bestpairing)
    print_result(bestpairing)
    return bestpairing

def print_result(list_or_pairing,p=p_shiny):
    if isinstance(list_or_pairing[0],str):
        print_result_list(list_or_pairing,p)
    if isinstance(list_or_pairing[0],tuple):
        print_result_pairing(list_or_pairing,p)

def print_result_pairing(dragonpairing,p=p_shiny):
    for val,pair in sorted([(breeding_value(d1,d2,p),(d1,d2)) for (d1,d2) in dragonpairing],reverse=True):
        print(f'{val:>9.4f}: {pair}')
    print(f'{breeding_sum_pairing(dragonpairing,p):.4f}  total mean' )
    print('='*80)
def print_result_list(dragonlist,p=p_shiny):
    print_result_pairing(list2pairs(dragonlist),p)
################################################################################

### SELECTION HEURISTICS #######################################################
# Now we need a good heuristic of how to select the 20 dragons to plug into the
# search_all() function, because nCr(190,20) is much too large (~5.5E26).

def dragons_by_score(score_func,p=p_shiny)->list[str]:
    return sorted(dragon_names,key=lambda d: score_func(d,p),reverse=True)

def print_scores(score_func,p=p_shiny):
    for d in dragons_by_score(score_func,p):
        print(f'{score_func(d,p):.4f}: {d}')

### Option 1 ###
# assign a "goodness-score" to each dragon
# just sum the egg values and factor in the shiny chance
# note that dragon_score(d1)+dragon_score(d2)!=breeding_value(d1,d2)
def dragon_score_1(dragon:str,p=p_shiny)->float:
    eggs = dragon_eggs(dragon)
    return p*0.5*dragon_value(dragon) + (1-p)*sum([egg_value(e) for e in eggs])/len(eggs)

def selection_1(n=20):
    return dragons_by_score(dragon_score_1)[:n]

### Option 2 ###
# order all possible pairs of dragons by their value (we already computed those),
# then pick the dragons of the first "free" n/2 pairs from the top

def selection_2(n=20):
    # [ (score,(d1,d2)),… ]
    pair_values = list(map(lambda pair: (lookup[pair],pair),itertools.combinations(dragon_names,2)))
    sl=[]
    for (v,(d1,d2)) in sorted(pair_values,reverse=True):
        if (d1 not in sl) and (d2 not in sl):
            sl.append(d1)
            sl.append(d2)
    return sl[:n]

### Option 3 ###
# almost the same as Option 2, but use the first n dragon from the top,
# even if that specific pair is not "free"
def selection_3(n=20):
    pair_values = list(map(lambda pair: (lookup[pair],pair),itertools.combinations(dragon_names,2)))
    sl=[]
    for (v,(d1,d2)) in sorted(pair_values,reverse=True):
        if (d1 not in sl):
            sl.append(d1)
        if (d2 not in sl):
            sl.append(d2)
    return sl[:n]

### Option 4 ###
# assign dragon score based on sum of all breeding values of pairs this dragon is part of
def selection_4(n=20):
    dragon_sums = {d:0 for d in dragon_names}
    pair_values = list(map(lambda pair: (lookup[pair],pair),itertools.combinations(dragon_names,2)))
    for (v,(d1,d2)) in pair_values:
        dragon_sums[d1]+=v
        dragon_sums[d2]+=v

    selection = [name for (score,name) in sorted([(dragon_sums[d],d) for d in dragon_sums],reverse=True)]
    return selection[:n]

################################################################################

### FUTURE WORK ################################################################
# There is still lots of room for improvement, some options that could be considered:
#
# * filter out duplicat pairings (dragons with identical egg groups), this can
#   cut down the search space significantly
# * search through more sets of dragons for a more confident best result, eg by:
#   - select 30 or so best individual dragons
#   - assign a score to each 20-combination (nCr(30,20)~3E7), based on an initial
#     small run of search_all()
#   - run a full search of the combination with best initial score
#
# * simulated annealing / hill climbing or genetic algorithm approach
# * rewrite in compiled language (C,Rust)
# * Parallelization (gen_pairings can easily be broken up into threads) and
# * Vectorization (eg. sum(lookup) could be matrix multiplication

