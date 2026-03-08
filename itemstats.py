#!/usr/bin/env python

from dataclasses import dataclass, field

import copy

import numpy as np

POWERUPS = ['heart', 'mushroom', 'fairy', 'eye', 'chilli', 'thunder', 'time', 'star', 'lucky', 'spoon', 'snow', 'ghost']
CHESTS = ['wood','iron','gold','diamond','pumpkin','present']
ITEMS = ['nothing', 'royal', 'chest', 'powerup', 'egg', 'portal', 'donut']

@dataclass
class ItemNode:
    key: str
    name: str = ''
    weight: int = 1
    # es macht Sinn, bei allen Items min/max crowns zu haben, weil einige skills/augments
    # aus einer festen Zahl einen zufälligen Wert machen (z.B. Riffallua)
    crowns_min: int = 0
    crowns_max: int = 0

    parent: ItemNode|None = None
    children: list[ItemNode] = field(default_factory=list) # = []


    def __post_init__(self):
        for child in self.children:
            child.parent = self

    def add_child(self, child: ItemNode):
        child.parent = self
        self.children.append(child)

    def level(self):
        if not self.parent:
            return 0
        else:
            return self.parent.level()+1

    def print_tree(self):
        print('  '*self.level() + f"ItemNode({self.key})")
        for child in self.children:
            child.print_tree()

    # get all leaf nodes only
    # useful for checking that sum(root_chances)==1
    def leaf_nodes(self):
        if not self.children:
            yield self
        else:
            for child in self.children:
                for leaf in child.leaf_nodes():
                    yield leaf


    def keys(self) -> list[str]:
        return [child.key for child in self.children]

    def find(self, key: str) -> ItemNode|None:
        if key == self.key:
            return self
        for child in self.children:
            result = child.find(key)
            if result:
                return result
            #return self.children[self.keys().index(key)]
        return None

    def weights_children(self) -> list[int]:
        return [child.weight for child in self.children]
    def weights_children_total(self) -> int:
        return sum(self.weights_children())

    def weights_siblings(self) -> int:
        if not self.parent:
            return 1
        return self.parent.weights_children_total()

    # probability relative to parent
    def rel_chance(self) -> float:
        if not self.parent:
            return 1 #root element
        return self.weight / self.parent.weights_children_total()

    # probability relative to root
    def root_chance(self) -> float:
        if not self.parent:
            return 1 #==root
        else:
            return self.rel_chance()*self.parent.root_chance()

    @property
    def crowns(self):# -> float:
        if not self.children:
            return (self.crowns_min + self.crowns_max)/2
        else:
            # rel_chance is correct, because crowns is recursive
            return sum( c.rel_chance()*c.crowns for c in self.children )

    @crowns.setter
    def crowns(self,value: int) -> None:
        self.crowns_min = value
        self.crowns_max = value

    
# U+2300 -> durchschnittszeichen ⌀
    def pprint(self) -> None:
        indent = '  '*self.level()
        w_rel = f"{self.weight:4f}/{self.weights_siblings():4f}" # because of strange percentages we may have non-integer weights
        p_rel = self.rel_chance()
        p_root = self.root_chance()
        ctext = f"({self.crowns_min}–{self.crowns_max}),⌀={self.crowns:.2f}c" if self.crowns else ' '*13
        c_rel = self.rel_chance() * self.crowns
        c_rel_percent = (
                1 if not self.parent else
                0 if self.parent.crowns == 0 else
                c_rel/self.parent.crowns*100 
                )
        #TODO c_abs_percent
        print(f"{indent+self.key:<13}: {ctext} {w_rel=}, {p_rel=:.4f}; {p_root=:.4f} => ~ {c_rel}={c_rel_percent}%")
        for child in self.children:
            child.pprint()


def base_items():
    return ItemNode('items', children= 
        [ ItemNode('nothing','Nothing',1000,0,0)
        , ItemNode('royal','Aristocrat',500,1,1)
        , ItemNode('chest','Chest',250, children=
            [ ItemNode('wood','Wooden Chest',50,1,3)
            , ItemNode('iron','Iron Chest',30,2,4)
            , ItemNode('gold','Golden Chest',15,3,5)
            , ItemNode('diamond','Diamond Chest',5,4,6)
            , ItemNode('pumpkin','Halloween Pumpkin',5,1,6)
            , ItemNode('present','Christmas Present',5,1,6)
            ] )
        , ItemNode('powerup','Powerup',150, children=
            [ ItemNode('heart','Dragonheart')
            , ItemNode('mushroom','Suspicious Mushroom')
            , ItemNode('fairy','Fairy Dust')
            , ItemNode('eye','Dragon Eye')
            , ItemNode('chilli','Dragon Chilli')
            , ItemNode('thunder','Thundercloud')
            , ItemNode('time','Sands of Time')
            , ItemNode('star','Cosmic Stars')
            , ItemNode('lucky','Lucky Charm')
            , ItemNode('spoon','Curved Spoon')
            , ItemNode('snow','Magic Snowflake')
            , ItemNode('ghost','Spooky Ghost')
            ] )
        , ItemNode('egg','Mystery Egg',25,0,0)
        , ItemNode('portal','Dimensional Rift',18,0,0)
        , ItemNode('donut','Donut',12,0,0)
        ] )


# construct a (n+1)x(n+1) matrix for a lucky streak of n towers
def construct_matrix(n:int, pul:float, pll:float) -> np.array:
    assert 0<=pul<=1
    assert 0<=pll<=1
    P = np.zeros((n+1,n+1))
    P[0,0] = 1-pul
    P[0,1] = pul
    P[n,0] = 1-pll
    for i in range(1,n+1):
        P[i,1] = pll
    for i in range(1,n):
        P[i,i+1] = 1-pll

    return P

# overall probability of being in a lucky state at any arbitrary tower
def compute_lucky(n,pul,pll):
    P = construct_matrix(n,pul,pll)

    # left eigenvector of P is right eigenvector of P.T
    (λs,vr) = np.linalg.eig(P.T)
    # alternatively we could use scipy for the left eigenvector
    #(λs,vl) = scipy.linalg.eig(P, left=True, right=False)
    assert np.isclose(λs[0].real,1)
    v = vr[:,0].real
    p = v/ sum(v) # because probabilities

    assert np.allclose(p @ P, p)
    return 1-p[0]

def parse_modifier(s: str|int|float, x: int|float) -> int|float:
    if isinstance(s,int) or isinstance(s,float):
        y = s
    elif s[0] == '*':
        y = float(s[1:]) * x
    elif s[-1] == '%':
        if s[0] == '+':
            y = x + float(s[1:-1])/100 * x
        elif s[0] == '-':
            y = x - float(s[1:-1])/100 * x
    else:
        y = float(s)
    return y

@dataclass
class Dragon:
    name: str
    speed: float = 1.0 #TODO???
    lucky_towers: int = 6 #TODO aus speed ausrechnen

    items: ItemNode = field(default_factory=base_items)


    weights: dict[str,str] = field(default_factory=dict)
    crowns: dict[str,str] = field(default_factory=dict)

    def __post_init__(self):
        for item in self.weights:
            self.modify_weight(item, self.weights[item])
        for item in self.crowns:
            self.modify_crowns(item, self.crowns[item])

    def modify_weight(self, item: str, modifier: str|int|float) -> Dragon:
        self.items.find(item).weight = parse_modifier(modifier, self.items.find(item).weight)
        return self

    def modify_crowns(self, item: str, modifier: str|int|float) -> Dragon:
        self.items.find(item).crowns = parse_modifier(modifier, self.items.find(item).crowns)
        return self

    def modify_weight_all_powerups(self, modifier) -> Dragon:
        for p in POWERUPS:
            self.modify_weight(p, modifier)
        return self
    def modify_crowns_all_powerups(self, modifier) -> Dragon:
        for p in POWERUPS:
            self.modify_crowns(p, modifier)
        return self

    def init_lucky_items(self):
        self.lucky_items = copy.deepcopy(self.items)
        self.lucky_items.find('nothing').weight *= 0
        self.lucky_items.find('chest').weight *= 5
        self.lucky_items.find('egg').weight *= 5
        self.lucky_items.find('donut').weight *= 5


    def pul(self):
        return self.items.find('lucky').root_chance()
    def pll(self):
        self.init_lucky_items()
        return self.lucky_items.find('lucky').root_chance()

    def lucky_time(self):
        return compute_lucky(self.lucky_towers, pul=self.pul(), pll=self.pll())

    def total(self,item) -> float:
        p = self.lucky_time()
        if item=='crowns':
            cu = self.items.crowns
            cl = self.lucky_items.crowns
            assert cl >= cu # alert in this special case
            return p*cl + (1-p)*cu
        else:
            pu = self.items.find(item).root_chance()
            pl = self.lucky_items.find(item).root_chance()
            assert pl >= pu #?
            return p*pl + (1-p)*pu # eggs/donuts/portals/royals only have amount 1

    def stats(self):
        print(f"=== {self.name} ===")
        self.items.pprint()
        print('%lucky =', self.lucky_time())
        print('crowns total:', self.total('crowns'))
        print('eggs   total:', self.total('egg'))
        print('donuts total:', self.total('donut'))
        print('--------------------')


base = Dragon('===BASE===')
#base.stats()

# speeds -> towers/12s
# 2.5 -> 7.5
# 3.0 -> 9.0
# 4.0 ->12.0
# f(x)=3x (floor)

#TODO some dragons may give _more_ crowns at a _lower_ level
level=10

koraline = Dragon('Koraline')
for chest in CHESTS:
    koraline.items.find(chest).crowns_max += 10
riffallua = Dragon('Riffallua')
riffallua.modify_weight('royal','-95%')
riffallua.items.find('royal').crowns_max = 100

dragons = [
base,
Dragon('Flappy', weights={'chest':'+100%'}),
koraline,
Dragon('Kimbaza', weights={'pumpkin':'+1000%'}),
Dragon('Rudfulf', weights={'present':'+1000%'}),
Dragon('Antaro', weights={'snow':'+1000%'}),
Dragon('Abymonio', weights={'ghost':'+1000%'}),
Dragon('Feulungo', crowns={'royal':3}),
Dragon('Yulung', crowns={'royal':3}),
Dragon('Bolgula', crowns={'royal':3}), # 1/2(top) + 5/2(bottom) = 6/2=3
Dragon('Cenixo', weights={'heart':'+1000%'}),
Dragon('Evererg').modify_weight('powerup','-7.5%').modify_weight_all_powerups(0).modify_weight('snow',1),
Dragon('Flarbonit').modify_weight('powerup','-7.5%').modify_weight_all_powerups(0).modify_weight('chilli',1),
Dragon('Kyrnyan', weights={'lucky':'+1000%'}),
Dragon('Gadah').modify_weight('fairy','+1000%'),
Dragon('Gerjileta').modify_weight('donut','+200%'),
Dragon('Ilvy').modify_crowns('royal',0),
Dragon('Iro').modify_weight('eye','+1000%'),
Dragon('Jaa Eda').modify_crowns_all_powerups(10), #TODO picking up *all* powerups may not always be optimal (eg. interrupting a lucky streak)
Dragon('Malame', weights={'powerup':'+100%', 'lucky':0}),
Dragon('Manyan').modify_weight('powerup','-7.5%').modify_weight_all_powerups(0).modify_weight('lucky',1),
Dragon('Leviaceanos').modify_weight('royal','+100000%'), #TODO check in game
Dragon('Memphara').modify_weight('time','+1000%'),
Dragon('Metreoloona').modify_weight('mushroom','+1000%'),
Dragon('Mormoza', crowns={'royal':2}),
Dragon('Tyranex', crowns={'royal':2}),
Dragon('Vetrece', crowns={'royal':14}),
Dragon('Munestra').modify_weight('egg','+100%'),
Dragon('Sunestro').modify_weight('royal','+100%'),
Dragon('Oad Lum').modify_weight('powerup','-7.5%').modify_weight_all_powerups(0).modify_weight('mushroom',1),
Dragon('Ohjo').modify_weight('powerup','-7.5%').modify_weight_all_powerups(0).modify_weight('eye',1),
Dragon('Ssenhid').modify_weight('powerup','-7.5%').modify_weight_all_powerups(0).modify_weight('heart',1),
Dragon('Zaguilza').modify_weight_all_powerups(0).modify_weight('eye',1),
Dragon('Orsonte', weights={'chest':'+200%', 'powerup':'-75%'}),
Dragon('Pepaga', crowns={'portal':20}),
Dragon('Quetzorotl', weights={'royal':'+50%', 'chest':'+50%', 'powerup':'+50%', 'egg':'+50%', 'donut':'+50%'}),
Dragon('R-D0N8T').modify_weight('donut','+100%'),
riffallua,
Dragon('Salozard').modify_weight('chilli','+1000%'),
Dragon('Salozora').modify_weight('spoon','+1000%'),
Dragon('Sel Doda').modify_weight('heart','+1000%'),
Dragon('X-K0SM0S').modify_weight('star','+1000%'),
Dragon('X-4L13N').modify_weight('portal','+10%'),
Dragon('Xen Lu').modify_weight('powerup','+100%'),
Dragon('Yon Lu 3', lucky_towers=3).modify_weight('powerup','+200%'), #TODO check actual value
Dragon('Yorochi').modify_weight('powerup','+1000%'),
#Dragon('Z-B0ND', lucky_towers=6+6*5*5), #TODO check actual value! 12s + 5* 60% + speedup from rifts
Dragon('Zinkireno').modify_weight('thunder','+1000%'),
]

# interesting observations:
# more powerups is often worse (eg. Yorochi)


by_crowns = sorted([(d.total('crowns'),d.name) for d in dragons], reverse=True)
by_eggs = sorted([(d.total('egg'),d.name) for d in dragons], reverse=True)
by_donuts = sorted([(d.total('donut'),d.name) for d in dragons], reverse=True)
assert len(by_crowns)==len(by_eggs)==len(by_donuts)


print('        CROWNS                              EGGS                                DONUTS')
for i in range(len(by_crowns)):
    print(f"\
        {by_crowns[i][0]:.5f} {by_crowns[i][1]:<20}\
        {by_eggs[i][0]:.5f} {by_eggs[i][1]:<20}\
        {by_donuts[i][0]:.5f} {by_donuts[i][1]:<20}\
        "
        )


base.stats()


### making (network) graphs:
# graphviz / dot
# dot2tex
# kgraphviewer
# python-graphviz ?
# python-pydot ?
# python-pygraphviz ?
# xdot ?

#
#def apply_multiplier(multiplier,weights):
#    return {i:(w*multiplier[i] if i in multiplier else w) for i,w in weights.items()}
#
#
#def dictmap(f, d):
#    return {k:f(v) for k,v in d.items()}
#def dictmult(m, d):
#    return dictmap(lambda x: m*x, d)
#

