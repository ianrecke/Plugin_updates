"""
FGES arrows utilities module.
"""

# Computer Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

SEP_NUMS = '_'
SEP_NS = ':'
SEP_SETS = ';'


def create_arrow(x, y, nayx, t, b):
    """Creates a string representation for an arrow."""

    nayx = SEP_NUMS.join((str(node) for node in nayx))
    t = SEP_NUMS.join((str(node) for node in t))
    return f'{x}{SEP_NUMS}{y}{SEP_NS}{nayx}{SEP_SETS}{t}', b


def get_vals(arrow):
    """Returns all the values for an arrow."""

    x, y = get_nodes(arrow)
    na_yx = get_nayx(arrow)
    t = get_t(arrow)
    bic = arrow[1]

    return x, y, na_yx, t, bic


def get_nodes(arrow):
    """Returns the nodes that form an arrow."""

    arr_str = arrow[0]
    nodes_str = arr_str.split(SEP_NS)[0]
    nodes = nodes_str.split(SEP_NUMS)

    return int(nodes[0]), int(nodes[1])


def get_origin(arrow):
    """Returns the origin node in the arrow."""

    arr_str = arrow[0]
    nodes_str = arr_str.split(SEP_NS)[0]
    x = nodes_str.split(SEP_NUMS)[0]

    return int(x)


def get_dest(arrow):
    """Returns the destination node in the arrow"""

    arr_str = arrow[0]
    nodes_str = arr_str.split(SEP_NS)[0]
    y = nodes_str.split(SEP_NUMS)[1]

    return int(y)


def get_nayx(arrow):
    """Returns the NaYX set in the arrow."""

    arr_str = arrow[0]
    sets_str = arr_str.split(SEP_NS)[1]
    na_yx_str = sets_str.split(SEP_SETS)[0]
    if len(na_yx_str) == 0:
        na_yx = set()
    else:
        na_yx = set([int(s) for s in na_yx_str.split(SEP_NUMS)])

    return na_yx


def get_t(arrow):
    """Returns the T set in the arrow."""

    arrow_str = arrow[0]
    sets_str = arrow_str.split(SEP_NS)[1]
    t_str = sets_str.split(SEP_SETS)[1]
    if len(t_str) == 0:
        return set({})
    else:
        return set([int(s) for s in t_str.split(SEP_NUMS)])


def get_bic(arrow):
    """Returns the bic score for the edge in the arrow."""
    return arrow[1]
