from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES_DICT = {}

PRIMITIVES_DICT['G1'] = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

PRIMITIVES_DICT['G2'] = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'csep_conv_3x3',
    'csep_conv_5x5',
    # 'cdil_conv_3x3',
    # 'cdil_conv_5x5'
]

PRIMITIVES_DICT['G3'] = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'csep_conv_3x3',
    'csep_conv_5x5',
    'csep3_conv_3x3',
    'csep3_conv_5x5',
]

PRIMITIVES_DICT['G3_2'] = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'csep_conv_3x3',
    'csep3_conv_3x3',
]

PRIMITIVES_DICT['G4'] = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'csep_conv_3x3',
    'csep_conv_5x5',
    'cdil_conv_3x3',
    'cdil_conv_5x5',
    'csep3_conv_3x3',
    'csep3_conv_5x5',
    'cdil3_conv_3x3',
    'cdil3_conv_5x5',
]

PRIMITIVES_DICT['G5'] = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'csep_conv_3x3',
    'csep_conv_5x5',
    'csep_conv_7x7',
    'cdil_conv_3x3',
    'cdil_conv_5x5',
    'cdil_conv_7x7',
    'csep3_conv_3x3',
    'csep3_conv_5x5',
    'csep3_conv_7x7',
    'cdil3_conv_3x3',
    'cdil3_conv_5x5',
    'cdil3_conv_7x7',
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)

AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

# DARTS = Genotype(normal=[('csep_conv_3x3', 0), ('csep_conv_3x3', 1), ('csep_conv_3x3', 0), ('csep_conv_3x3', 1), ('csep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('cdil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
