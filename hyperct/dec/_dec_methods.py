"""
Standalone DEC (Discrete Exterior Calculus) functions for hyperct.

Provides Clifford algebra integration for discrete exterior calculus
operations (sharp/flat musical isomorphisms).

These functions operate on a Complex instance passed as the `hc` parameter
(where applicable), rather than being methods on the class itself.
"""
import logging

import numpy

# Optional clifford dependency
try:
    import clifford as cf
    _clifford_available = True
except ImportError:
    logging.warning("Discrete exterior calculus functionality will be "
                    "unavailable. To use, install the clifford package with "
                    "`pip install clifford`")
    _clifford_available = False


def clifford(hc, dim, q=''):
    """
    Memoize a specified clifford algebra so that it is only needed to
    initialise once, default is euclidean Cl(dim)
    :param hc: Complex instance
    :param dim:
    :param q:
    :return:
    """
    if not _clifford_available:
        logging.warning("Discrete exterior calculus functionality will be "
                        "unavailable. To use, install the clifford package"
                        " with `pip install clifford`")
        return None
    try:
        return hc.calgebras[str(dim) + q]
    except (AttributeError, KeyError):
        if not hasattr(hc, 'calgebras'):
            hc.calgebras = {}
        layout, blades = cf.Cl(dim)
        i = 0
        one_forms = []
        for n in blades:
            i += 1
            one_forms.append(1 * blades[n])
            if i == dim:
                break

        hc.calgebras[str(dim) + q] = layout, blades, one_forms
        return hc.calgebras[str(dim) + q]


def sharp(hc, v_x):
    """
    Convert a vector to a 1-form

    TODO: Should be able to convert k-forms

    :param hc: Complex instance
    :param v_x:  vector, dimension may differ for class hc.dim
    :return:
    """
    dim = len(v_x)
    # Call memoized Clifford algebra
    result = clifford(hc, dim)
    if result is None:
        return None
    layout, blades, one_forms = result
    #NOTE: Using numpy.dot(v_x, one_forms) converts one_forms
    form = 0
    for i, of in enumerate(one_forms):
        form += v_x[i] * of
    return form


def flat(form, dim):
    """
    Convert a 1-form to a vector

    Note that dim can by computed by evaluating len(v_x) after iterating
    form, but this is probably expensive.

    :param form:
    :param dim:
    :return: v_x, numpy array
    """
    # Find 1 form grade projection
    oneform = form(1)  # Not reducing struction
    v_x = []
    for f in form:
        v_x.append(f)
    v_x = numpy.array(v_x)
    v_x = v_x[1:dim+1]
    return v_x
