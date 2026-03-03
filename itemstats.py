#!/usr/bin/env python

from dataclasses import dataclass, field
import abc

# abstract class for type hints / interface. just for practice
@dataclass
class AbstractItem(abc.ABC):
    key: str
    name: str = ''
    weight: int = 1

    @property
    @abc.abstractmethod
    def crowns(self) -> float:
        ...

# es macht Sinn, bei allen Items min/max crowns zu haben, weil einige skills/augments
# aus einer festen Zahl einen zufälligen Wert machen (z.B. Riffallua)
@dataclass
class Item(AbstractItem):
    crowns_min: int = 0
    crowns_max: int = 0

    @property
    def crowns(self) -> float:
        return (self.crowns_min + self.crowns_max)/2

@dataclass
class MetaItem(AbstractItem):
    items: list[AbstractItem] = field(default_factory=list) # = []

    @property
    def crowns(self) -> float:
        # rel_chance is correct, because crowns is recursive
        return sum( self.rel_chance(i)*i.crowns for i in self.items )

    def keys(self) -> list[str]:
        return [item.key for item in self.items]

    def get(self, key: str) -> AbstractItem:
        return self.items[self.keys().index(key)]

    def weights(self) -> list[int]:
        return [item.weight for item in self.items]
    def totalweight(self) -> int:
        return sum(self.weights())

    def rel_chance(self,item: AbstractItem) -> float:
        return item.weight / self.totalweight()

    def abs_chance(self,item: AbstractItem,p_parent=1) -> float:
        if isinstance(item,Item):
            pass
        if isinstance(item,MetaItem):
            pass


    def pprint(self,level=0,p_parent=1) -> None:
        for item in self.items:
            indent = '  '*level
            w_rel = f"{item.weight}/{self.totalweight()}"
            p_rel = self.rel_chance(item)
            p_abs = p_rel * p_parent
            print(f"{indent}{item.key}: {w_rel=}, {p_rel=}; {p_abs=} => {item.crowns}")
            if isinstance(item,MetaItem) :
                item.pprint(level+1,p_parent=p_abs)

@dataclass
class Dragon:
    name: str

    itemroot = MetaItem('items', items= 
        [ Item('nothing','Nothing',1000,0,0)
        , Item('royal','Aristocrat',500,1,1)
        , MetaItem('chest','Chest',250, items=
            [ Item('wood','Wooden Chest',50,1,3)
            , Item('iron','Iron Chest',30,2,4)
            , Item('gold','Golden Chest',15,3,5)
            , Item('diamond','Diamond Chest',5,4,6)
            , Item('pumpkin','Halloween Pumpkin',5,1,6)
            , Item('present','Christmas Present',5,1,6)
            ] )
        , MetaItem('powerup','Powerup',150, items=
            [ Item('heart','Dragonheart')
            , Item('mushroom','Suspicious Mushroom')
            , Item('fairy','Fairy Dust')
            , Item('eye','Dragon Eye')
            , Item('chilli','Dragon Chilli')
            , Item('thunder','Thundercloud')
            , Item('time','Sands of Time')
            , Item('star','Cosmic Stars')
            , Item('luck','Lucky Charm')
            , Item('spoon','Curved Spoon')
            , Item('snow','Magic Snowflake')
            , Item('ghost','Spooky Ghost')
            ] )
        , Item('egg','Mystery Egg',25,0,0)
        , Item('portal','Dimensional Rift',18,0,0)
        , Item('donut','Donut',12,0,0)
        ] )

d = Dragon('base')
d.itemroot.pprint()
print(d.itemroot.crowns)

test = MetaItem('items',items=
    [ Item('1')
    , MetaItem('2', items=
        [ Item('2.1')
        , MetaItem('2.2', items=
            [ Item('2.2.1')
            , Item('2.2.2')
            ]
            )
        ]
        )
    ]
    )

#multiplier_lucky = {
#    'nothing': 0,
#    'chest': 5,
#    'egg': 5,
#    'donut': 5
#}
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
#
## average crowns per chest:
#chest_avg = { i:((c[0]+c[1])/2) for i,c in chest_crowns.items() }
#
## total chest average:
#chest_total = sum( chest_avg[i]*chances(chest_weights)[i] for i in chest_weights )
#
## normal return value per 100 towers
#xn = chances(main_weights)['royal']*1 + chances(main_weights)['chest']*chest_total
#print(xn*100)
## lucky:
#xl = chances(apply_multiplier(multiplier_lucky,main_weights))['royal']*1 + chances(apply_multiplier(multiplier_lucky,main_weights))['chest']*chest_total
#print(xl*100)

