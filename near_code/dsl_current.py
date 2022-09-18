import dsl


DSL_DICT = {
    ('atom', 'atom') : [dsl.SimpleITE, dsl.FullInputAffineFunction]
}

CUSTOM_EDGE_COSTS = {
    ('list', 'list') : {},
    ('list', 'atom') : {},
    ('atom', 'atom') : {}
}
