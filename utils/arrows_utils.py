SEP_NUMS = '_'
SEP_I_J = ':'
SEP_SETS = ';'


def create_arrow_str(arrow_input):
    x = arrow_input[0][0]
    y = arrow_input[0][1]
    na_yx = list(arrow_input[1])
    t = list(arrow_input[2])
    bic = arrow_input[3]

    arrow = (
        str(x) + SEP_NUMS + str(y) + SEP_I_J +  # i, j
        SEP_NUMS.join((str(node) for node in na_yx)) + SEP_SETS +  # set(na_yx)
        SEP_NUMS.join((str(node) for node in t)),  # set(t)
        bic  # BIC score buff
    )

    return arrow


def get_all_vals(arrow):
    x = get_i(arrow)
    y = get_j(arrow)
    na_yx = get_na_yx(arrow)
    t = get_t(arrow)
    bic = get_bic(arrow)

    return x, y, na_yx, t, bic


def get_arrow_str(arrow):
    return arrow[0]


def get_bic(arrow):
    return arrow[1]


def get_i_j_str(input_arr):
    i_j_str = input_arr.split(SEP_I_J)[0]

    return i_j_str


def get_sets_str(input_arr):
    sets_str = input_arr.split(SEP_I_J)[1]

    return sets_str


def get_i_j(arrow):
    i = get_i(arrow)
    j = get_j(arrow)

    return i, j


def get_i(arrow):
    input_arr = get_arrow_str(arrow)
    i_j_str = get_i_j_str(input_arr)
    i = i_j_str.split(SEP_NUMS)[0]

    return int(i)


def get_j(arrow):
    input_arr = get_arrow_str(arrow)
    i_j_str = get_i_j_str(input_arr)
    j = i_j_str.split(SEP_NUMS)[1]

    return int(j)


def get_na_yx(arrow):
    input_arr = get_arrow_str(arrow)
    sets_str = get_sets_str(input_arr)
    na_yx_str = sets_str.split(SEP_SETS)[0]
    if len(na_yx_str) == 0:
        na_yx = set({})
    else:
        na_yx = set([int(s) for s in na_yx_str.split(SEP_NUMS)])

    return na_yx


def get_t(arrow):
    input_arr = get_arrow_str(arrow)
    sets_str = get_sets_str(input_arr)
    t_str = sets_str.split(SEP_SETS)[1]
    if len(t_str) == 0:
        t = set({})
    else:
        t = set([int(s) for s in t_str.split(SEP_NUMS)])

    return t
