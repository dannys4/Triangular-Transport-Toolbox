import numpy as np
import copy

def merge_cached_variables(key, old_value, add_value, polyfunc_str):
    coeffs, new_coeffs = copy.copy(old_value[0]), copy.copy(add_value[0])
    new_coeffs = new_coeffs[:,-1]
    max_degree = coeffs.shape[0] - 1
    new_degree = len(new_coeffs) - 1
    if new_degree < max_degree:
        new_coeffs = np.pad(new_coeffs, (0, max_degree - new_degree))
    is_order_cached = coeffs.shape[1] > new_degree and np.all(coeffs[:,new_degree-1] == new_coeffs)
    # Save them in column ct[i] if not (and expand the matrix as necessary)
    if not is_order_cached:
        if max_degree < new_degree:
            coeffs = np.vstack(
                [coeffs, np.zeros((new_degree - max_degree, coeffs.shape[1]))])
            coeffs = np.hstack([coeffs, np.zeros((coeffs.shape[0], new_degree - coeffs.shape[1]))])
        coeffs[:,new_degree-1] = new_coeffs
    coeffs_str = np.array2string(coeffs, separator=',', max_line_width=np.inf).replace('\n','')
    dimension = key.split("_")[-1]
    eval_str = polyfunc_str + "(__x__[...,"+str(dimension)+"],"+coeffs_str+")"
    return (coeffs, eval_str)

def modifier_log_family(family_key, modifier_log, polyfunc_str, new_coefficients, dimension, order):
    coeffs = None
    curr_degree = len(new_coefficients)
    # Save the variable's family
    if family_key in modifier_log["cached_variables"]:
        # If we have saved the family before, get the coefficients
        coeffs, _ = modifier_log["cached_variables"][family_key]
        # Check if we've already saved these coefficients already
        is_order_cached = coeffs.shape[1] >= order and coeffs[:,order] == new_coefficients
        # Save them in column ct[i] if not (and expand the matrix as necessary)
        if not is_order_cached:
            max_degree = coeffs.shape[0]
            if max_degree < curr_degree:
                coeffs = np.vstack(
                    [coeffs, np.zeros((curr_degree - max_degree, coeffs.shape[1]))])
            elif max_degree > curr_degree:
                new_coefficients = np.pad(
                    new_coefficients, (0, max_degree - curr_degree))
            if coeffs.shape[1] < order:
                coeffs = np.hstack([coeffs, np.zeros((coeffs.shape[0], order - coeffs.shape[1]))])
            coeffs[:,order-1] = new_coefficients
    else:

        coeffs = np.zeros((order+1, order))
        coeffs[:,order-1] = new_coefficients

    coeffs_str = np.array2string(coeffs, separator=',', max_line_width=np.inf).replace('\n','')
    eval_str = polyfunc_str + \
        "(__x__[...,"+str(dimension)+"],"+coeffs_str+")"
    modifier_log["cached_variables"][family_key] = (coeffs, eval_str)

def write_polynomial_basis_function(term, polyfunc, polyfunc_str, polyfunc_der, mode='standard', k=None):
    """
    This function assembles a string for a specific term of the map
    component functions. This can be a polynomial, a Hermite function, a
    radial basis function, or similar.

    Variables:

        term
            [variable] : either an empty list, a list, or a string, which
            specifies what function type this term is supposed to be.
            Empty lists correspond to a constant, lists specify polynomials
            or Hermite functions, and strings denote special terms such as
            radial basis functions.

        mode - [default = 'standard']
            [string] : a keyword which defines whether the term's string
            returned should be conventional ('standard') or its derivative
            ('derivative').

        k - [default = None]
            [integer or None] : an integer specifying what dimension of the
            samples the 'term' corresponds to. Used to clarify with respect
            to what variable we take the derivative.
    """

    import copy

    # First, check for input errors ---------------------------------------

    # Check if mode is valid
    if mode not in ['standard', 'derivative']:
        raise ValueError(
            "Mode must be either 'standard' or 'derivative'. Unrecognized mode: "+str(mode))

    # If derivative, check if k is specified
    if mode == 'derivative' and k is None:
        raise ValueError(
            "If mode == 'derivative', specify an integer for k to inform with regards to which variable we take the derivative. Currently specified: k = "+str(k))

    # If derivative, check if k is an integer
    if mode == 'derivative' and type(k) is not int:
        raise ValueError(
            "If mode == 'derivative', specify an integer for k to inform with regards to which variable we take the derivative. Currently specified: k = "+str(k))

    # Initiate the modifier log -------------------------------------------

    # This variable returns information about whether there is anything
    # special about this term. If this is not None, it is a dictionary with
    # the following possible keys:
    #   "constant"  this is a constant term
    #   "ST"        this is a RBF-based special term
    #   "HF"        this is a polynomial with a Hermite function modifier
    #   "LIN"       this is a polynomial with a linearization modifier
    #   "HFLIN"     this is a polynomial with both modifiers

    modifier_log = {}

    # -----------------------------------------------------------------
    # Check for modifiers
    # -----------------------------------------------------------------

    # Check for Hermite function modifier
    # TRUE if modifier is active, else FALSE
    hermite_function_modifier = np.asarray(
        [i == 'HF' for i in term],
        dtype=bool).any()

    # Check for linearization modifier
    # TRUE if modifier is active, else FALSE
    linearize = np.asarray(
        [i == 'LIN' for i in term],
        dtype=bool).any()

    # Remove all string-based modifiers
    term = [i for i in term if type(i) != str]

    # -----------------------------------------------------------------
    # Construct the polynomial term
    # -----------------------------------------------------------------

    # Extract the unique entries and their counts
    ui, ct = np.unique(term, return_counts=True)
    modifier_log

    # Both Hermite function and linearization modifiers are active
    if hermite_function_modifier and linearize:

        # Log this term
        modifier_log = {
            "HFLIN": None}

    # Hermite function modifiers is active
    elif hermite_function_modifier:

        # Log this term
        modifier_log = {
            "HF": None}

    # Linearization modifiers is active
    elif linearize:

        # Log this term
        modifier_log = {
            "LIN": None}

    # Add a "variables" key to the modifier_log, if it does not already exist
    if "variables" not in list(modifier_log.keys()):
        modifier_log["variables"] = {}

    if "cached_variables" not in list(modifier_log.keys()):
        modifier_log["cached_variables"] = {}

    # Create an empty string
    string = ""

    # Go through all unique entries
    for i in range(len(ui)):

        # Create an array of polynomial coefficients
        dummy_coefficients = np.array([0.]*ct[i] + [1.])

        # Normalize the influence of Hermite functions
        if hermite_function_modifier:

            # Evaluate a naive Hermite function
            hf_x = np.linspace(-100, 100, 100001)
            hfeval = polyfunc(dummy_coefficients)(hf_x)*np.exp(-hf_x**2/4)

            # Scale the polynomial coefficient to normalize its maximum value
            dummy_coefficients[-1] = 1/np.max(np.abs(hfeval))

        # -------------------------------------------------------------
        # Standard polynomial
        # -------------------------------------------------------------

        if mode == 'standard' or (mode == 'derivative' and ui[i] != k):

            # Create a variable key
            family_key = "P_"+str(ui[i])

            modifier_log_family(family_key, modifier_log, polyfunc_str, dummy_coefficients, ui[i], ct[i])

            key = family_key+"_O_"+str(ct[i])
            if hermite_function_modifier:
                key += "_HF"
            if linearize:
                key += "_LIN"

            # Set up function -----------------------------------------

            # Extract the polynomial
            var = copy.copy(family_key)

            # Get the index of these coefficients
            order_idx = ct[i] - 1
            # Access the correct order polynomial slice
            var += "[" + str(order_idx) + ",...]"

            # Save the variable ---------------------------------------
            if key not in list(modifier_log["variables"].keys()):
                modifier_log["variables"][key] = copy.copy(var)

            # Add the variable to the string --------------------------
            string += copy.copy(key)

            # Add a multiplier, in case there are more terms
            string += " * "

            # Add Hermite function ------------------------------------
            if hermite_function_modifier:
                HF_mod_key = "HFM_"+str(ui[i])
                if HF_mod_key not in list(modifier_log["variables"].keys()):
                    HF_mod_var = "np.exp(-__x__[...,"+str(ui[i])+"]**2/4)"
                    modifier_log["variables"][HF_mod_key] = HF_mod_var
                string += copy.copy(HF_mod_key)
                string += " * "

        # -------------------------------------------------------------
        # Derivative of polynomial
        # -------------------------------------------------------------

        elif mode == 'derivative':
            # Find the derivative coefficients
            dummy_coefficients_der = polyfunc_der(dummy_coefficients)

            # Create a variable key
            family_key_der = "P_"+str(ui[i])+"_DER"
            modifier_log_family(family_key_der, modifier_log, polyfunc_str, dummy_coefficients_der, ui[i], ct[i])

            # Create a variable key for the standard polynomial
            key = "P_"+str(ui[i])+"_O_"+str(ct[i])

            # Create a variable key for its derivative
            keyder = "P_"+str(ui[i])+"_O_"+str(ct[i])+"_DER"

            # Set up function -----------------------------------------

            # Extract the polynomial
            varder = copy.copy(family_key_der)

            # Get the index of the coefficients
            order_idx = ct[i] - 1

            # Access the correct order polynomial slice
            varder += "[" + str(order_idx) + ",...]"

            # Save the variable ---------------------------------------
            if keyder not in list(modifier_log["variables"].keys()):
                modifier_log["variables"][keyder] = copy.copy(varder)

            # Add the variable to the string --------------------------
            if not hermite_function_modifier:
                string += copy.copy(varder)

            # Add Hermite function ------------------------------------

            # https://www.wolframalpha.com/input/?i=derivative+of+f%28x%29*exp%28-x%5E2%2F4%29+wrt+x

            if hermite_function_modifier:

                # If we have a hermite function modifier, we also need
                # the original form of the polynomial

                # Set up function -------------------------------------
                
                # Get the polynomial family
                family_key = "P_"+str(ui[i])

                # Make sure we calculate the polynomial family
                modifier_log_family(family_key, modifier_log, polyfunc_str, dummy_coefficients, ui[i], ct[i])

                # Extract the polynomial
                varbase = copy.copy(family_key)

                # Get the index of these coefficients
                order_idx = ct[i] - 1
                # Access the correct order polynomial slice
                varbase += "[" + str(order_idx) + ",...]"

                # Save the variable -----------------------------------
                if key not in list(modifier_log["variables"].keys()):
                    modifier_log["variables"][key] = copy.copy(varbase)

                # Add Hermite Function Modifier to precalc if not there already
                HF_mod_key = "HFM_"+str(ui[i])
                if HF_mod_key not in list(modifier_log["variables"].keys()):
                    HF_mod_var = "np.exp(-__x__[...,"+str(ui[i])+"]**2/4)"
                    modifier_log["variables"][HF_mod_key] = HF_mod_var

                # Now we can construct the actual derivative ----------
                string = "-1/2*"+HF_mod_key + \
                    "*(__x__[...,"+str(ui[i])+"]*"+key+" - 2*"+keyder+")"

            # Add a multiplier, in case there are more terms ----------
            string += " * "

    # Remove the last multiplier " * "
    string = string[:-3]

    # If the variable we take the derivative against is not in the term,
    # overwrite the string with zeros
    if mode == 'derivative' and k not in ui:

        # Overwrite string with zeros
        string = "np.zeros(__x__.shape[:-1])"

    return string, modifier_log

def function_constructor_alternative(self, k=None):
    """
    This function assembles the string for the monotone and nonmonotone map
    components, then converts these strings into functions.

    Variables:

        k - [default = None]
            [integer or None] : an integer specifying what dimension of the
            samples the 'term' corresponds to. Used to clarify with respect
            to what dimension we build this basis function
    """

    import numpy as np
    import copy

    if k is None:

        # Do we only construct the functions for one dimension?
        partial_construction = False

        # Construct the functions for all dimensions
        Ks = np.arange(self.D)

        # Initialize empty lists for the monotone part functions, their
        # corresponding strings, and coefficients.
        self.fun_mon = []
        self.fun_mon_strings = []
        self.coeffs_mon = []

        # Initialize empty lists for the nonmonotone part functions, their
        # corresponding strings, and coefficients.
        self.fun_nonmon = []
        self.fun_nonmon_strings = []
        self.coeffs_nonmon = []

        # Check for any special terms
        self.check_for_special_terms()
        self.determine_special_term_locations()

    elif np.isscalar(k):

        # Do we only construct the functions for one dimension?
        partial_construction = True

        # Construct the functions only for this dimension
        Ks = [k]

    else:

        # Input is not recognized. Raise an error.
        raise Exception(
            "'k' for function_constructor_alternative must be either None or an integer.")

    # Go through all terms
    for k in Ks:

        # =================================================================
        # =================================================================
        # Step 1: Build the monotone function
        # =================================================================
        # =================================================================

        # Define modules to load
        modules = ["import numpy as np", "import copy"]

        # =================================================================
        # Extract the terms
        # =================================================================

        # Define the terms composing the transport map component
        terms = []

        # Prepare a counter for the special terms
        ST_counter = np.zeros(self.X.shape[-1], dtype=int)

        # Prepare a dictionary for precalculated variables
        dict_precalc = {}

        # Mark which of these are special terms, in case we want to create
        # permutations of multiple RBFS
        ST_indices = []

        # Go through all terms
        for i, entry in enumerate(self.monotone[k]):

            # -------------------------------------------------------------
            # Convert the map specification to a function
            # -------------------------------------------------------------

            # Find the term's function
            term, modifier_log = self.write_basis_function(
                term=entry,
                mode='standard')

            # -------------------------------------------------------------
            # Extract any precalculations, where applicable
            # -------------------------------------------------------------

            # If this term includes and precalculations, extract them
            if "variables" in list(modifier_log.keys()):

                # There are precalculating variables. Go through each
                for key in list(modifier_log["variables"].keys()):

                    # Have we logged this one already?
                    if key not in list(dict_precalc.keys()):

                        # No, we haven't. Add it.
                        dict_precalc[key] = copy.copy(
                            modifier_log["variables"][key]).replace("__x__", "x")

                        # Wait a moment! Are we linearizing this term?
                        if key.endswith("_LIN"):

                            # Yes, we are! What dimension is this?
                            d = int(copy.copy(key).split("_")[1])

                            # Edit the term
                            dict_precalc[key] = \
                                copy.copy(dict_precalc[key]).replace("__x__", "x_trc") + " * " + \
                                "(1 - vec[:,"+str(d)+"]/"+str(self.linearization_increment)+") + " + \
                                copy.copy(dict_precalc[key]).replace("__x__", "x_ext") + " * " + \
                                "vec[:,"+str(d)+"]/" + \
                                str(self.linearization_increment)

            # -------------------------------------------------------------
            # Post-processing for special terms
            # -------------------------------------------------------------

            # Is this term a special term?
            if "ST" in list(modifier_log.keys()):

                # Mark this term as a special one
                ST_indices.append(i)

                # Yes, it is. Add additional modules to load, if necessary
                if "import scipy.special" not in modules:
                    modules     .append("import scipy.special")

                # Extract this special term's dimension
                idx = modifier_log["ST"]

                # Is this a cross-term?
                # Cross-terms are stored in a separate key; access it, if
                # necessary.
                if k+self.skip_dimensions != idx:
                    # Yes, it is.
                    ctkey = "['cross-terms']"
                else:
                    # No, it isn't.
                    ctkey = ""

                # Replace __mu__ with the correct ST location variable
                term = term.replace(
                    "__mu__",
                    "self.special_terms["+str(k+self.skip_dimensions)+"]"+ctkey+"["+str(idx)+"]['centers']["+str(ST_counter[idx])+"]")

                # Replace __scale__ with the correct ST location variable
                # self.special_terms[k][d]

                term = term.replace(
                    "__scale__",
                    "self.special_terms["+str(k+self.skip_dimensions)+"]"+ctkey+"["+str(idx)+"]['scales']["+str(ST_counter[idx])+"]")

                # Increment the special term counter
                ST_counter[idx] += 1

            # -------------------------------------------------------------
            # Add the term to the list
            # -------------------------------------------------------------

            # If any dummy __x__ remain, replace them
            term = term.replace("__x__", "x")

            # Store the term
            terms   .append(copy.copy(term))

        # -----------------------------------------------------------------
        # If there are special cross-terms, create them
        # -----------------------------------------------------------------

        # Are there multiple special terms?
        # if np.sum([True if x != k else False for x in list(self.special_terms[k+self.skip_dimensions].keys())]) > 1:
        # if np.sum([True if x != 0 else False for x in self.RBF_counter_m[k,:]]) > 1:
        if 'cross-terms' in list(self.special_terms[k+self.skip_dimensions].keys()):

            import itertools

            # Yes, there are multiple special terms. Extract these terms.
            RBF_terms = [terms[i] for i in ST_indices]

            # Check what variables these terms are affiliated with
            RBF_terms_dim = - np.ones(len(RBF_terms), dtype=int)
            for ki in range(k+1+self.skip_dimensions):
                for i, term in enumerate(RBF_terms):
                    if "x[...,"+str(ki)+"]" in term:
                        RBF_terms_dim[i] = ki
            RBF_terms_dims = np.unique(np.asarray(RBF_terms_dim))

            # Create a dictionary with the different terms
            RBF_terms_dict = {}
            for i in RBF_terms_dims:
                RBF_terms_dict[i] = [RBF_terms[j] for j in range(
                    len(RBF_terms)) if RBF_terms_dim[j] == i]

            # Create all combinations of terms
            RBF_terms_grid = copy.deepcopy(RBF_terms_dict[RBF_terms_dims[0]])
            for i in RBF_terms_dims[1:]:

                # Create a grid with the next dimension
                RBF_terms_grid = list(itertools.product(
                    RBF_terms_grid,
                    copy.deepcopy(RBF_terms_dict[i])))

                # Convert this list of tuples into a new list of strings
                RBF_terms_grid = \
                    [entry[0]+"*"+entry[1] for entry in RBF_terms_grid]

            # Now remove all original RBF terms
            terms = [entry for i, entry in enumerate(
                terms) if i not in ST_indices]

            # Now add all the grid terms
            terms += RBF_terms_grid

        # -----------------------------------------------------------------
        # Add monotone coefficients
        # -----------------------------------------------------------------

        if not partial_construction:
            # Append the parameters
            self.coeffs_mon     .append(np.ones(len(terms))*self.coeffs_init)

        # =================================================================
        # Assemble the monotone function
        # =================================================================

        # Prepare the basis string
        string = "def fun(x,self):\n\t\n\t"

        # -----------------------------------------------------------------
        # Load module requirements
        # -----------------------------------------------------------------

        for entry in modules:
            string += copy.copy(entry)+"\n\t"
        string += "\n\t"  # Another line break for legibility

        # -----------------------------------------------------------------
        # Prepare linearization, if necessary
        # -----------------------------------------------------------------

        # If linearization is active, truncate the input x
        if self.linearization is not None:

            # First, find our which parts are outside the linearization hypercube
            string += "vec_below = copy.copy(x) - self.linearization_threshold[:,0][np.newaxis,:];\n\t"
            # Set all values above to zero
            string += "vec_below[vec_below >= 0] = 0;\n\t"
            string += "vec_above = copy.copy(x) - self.linearization_threshold[:,1][np.newaxis,:];\n\t"
            # Set all values below to zero
            string += "vec_above[vec_above <= 0] = 0;\n\t"
            string += "vec = vec_above + vec_below;\n\t"

            # Then convert the two arrays to boolean markers
            # Find all particles BELOW the lower linearization band
            string += "below = (vec_below < 0);\n\t"
            # Find all particles ABOVE the upper linearization band
            string += "above = (vec_above > 0);\n\t"
            # This is a matrix where all entries outside the linearization bands are 1 and all entries inside are 0
            string += "shift = np.asarray(below,dtype=float) + np.asarray(above,dtype=float);\n\t"

            # Truncate all values outside the hypercube
            string += "x_trc = copy.copy(x);\n\t"
            string += "for d in range(x.shape[1]):\n\t\t"
            # All values below the linearization band of this dimension are snapped to its border
            string += "x_trc[below[:,d],d] = self.linearization_threshold[d,0];\n\t\t"
            # All values above the linearization band of this dimension are snapped to its border
            string += "x_trc[above[:,d],d] = self.linearization_threshold[d,1];\n\t"

            # Add a space to the next block
            string += "\n\t"

            # Also crate an extrapolated version of x_trc
            string += "x_ext = copy.copy(x_trc);\n\t"
            # Offset all values which have been snapped by a small increment
            string += "x_ext += shift*" + \
                str(self.linearization_increment)+";\n\t"

            # Add a space to the next block
            string += "\n\t"

        # -----------------------------------------------------------------
        # Prepare precalculated variables
        # -----------------------------------------------------------------

        # Add all precalculation terms
        for key in list(dict_precalc.keys()):

            string += key + " = " + copy.copy(dict_precalc[key]) + ";\n\t"

        # -----------------------------------------------------------------
        # Assemble function output
        # -----------------------------------------------------------------

        # Prepare the result string
        if len(terms) == 1:  # Only a single term, no need for stacking

            string += "result = "+copy.copy(terms[0])+"[:,np.newaxis];\n\t\n\t"

        else:  # If we have more than one term, start stacking the result

            # Prepare the stack
            string += "result = np.stack((\n\t\t"

            # Go through each entry in terms, add them one by one
            for entry in terms:
                string += copy.copy(entry) + ",\n\t\t"

            # Remove the last ",\n\t\t" and close the stack
            string = string[:-4]
            string += "),axis=-1)\n\t\n\t"

        # Return the result
        string += "return result"

        # -----------------------------------------------------------------
        # Finish function construction
        # -----------------------------------------------------------------

        if not partial_construction:

            # Append the function string
            self.fun_mon_strings    .append(string)

            # Create an actual function
            funstring = "fun_mon_"+str(k)
            exec(string.replace("fun", funstring), globals())
            exec("self.fun_mon.append(copy.deepcopy("+funstring+"))")

        else:

            # Insert the function string
            self.fun_mon_strings[k] = copy.copy(string)

            # Create an actual function
            funstring = "fun_nonmon_"+str(k)
            exec(string.replace("fun", funstring), globals())
            exec("self.fun_mon[k] = copy.deepcopy("+funstring+")")

        # =================================================================
        # =================================================================
        # Step 2: Build the nonmonotone function
        # =================================================================
        # =================================================================

        if not partial_construction:

            # Append the parameters
            self.coeffs_nonmon  .append(
                np.ones(len(self.nonmonotone[k]))*self.coeffs_init)

        # Define modules to load
        modules = ["import numpy as np", "import copy"]

        # =================================================================
        # Extract the terms
        # =================================================================

        # Define the terms composing the transport map component
        terms = []

        # Prepare a counter for the special terms
        ST_counter = np.zeros(self.X.shape[-1], dtype=int)

        # Prepare a dictionary for precalculated variables
        dict_precalc = {}

        # Go through all terms
        for entry in self.nonmonotone[k]:

            # -------------------------------------------------------------
            # Convert the map specification to a function
            # -------------------------------------------------------------

            # Find the term's function
            term, modifier_log = self.write_basis_function(
                term=entry,
                mode='standard')

            # -------------------------------------------------------------
            # Extract any precalculations, where applicable
            # -------------------------------------------------------------

            # If this term includes and precalculations, extract them
            if "variables" in list(modifier_log.keys()):

                # There are precalculating variables. Go through each
                for key in list(modifier_log["variables"].keys()):

                    # Have we logged this one already?
                    if key not in list(dict_precalc.keys()):

                        # No, we haven't. Add it.
                        dict_precalc[key] = copy.copy(
                            modifier_log["variables"][key]).replace("__x__", "x")

                        # Wait a moment! Are we linearizing this term?
                        if key.endswith("_LIN"):

                            # Yes, we are! What dimension is this?
                            d = int(copy.copy(key).split("_")[1])

                            # Edit the term
                            dict_precalc[key] = \
                                copy.copy(dict_precalc[key]).replace("__x__", "x_trc") + " * " + \
                                "(1 - vec[:,"+str(d)+"]/"+str(self.linearization_increment)+") + " + \
                                copy.copy(dict_precalc[key]).replace("__x__", "x_ext") + " * " + \
                                "vec[:,"+str(d)+"]/" + \
                                str(self.linearization_increment)

            # -------------------------------------------------------------
            # Post-processing for special terms
            # -------------------------------------------------------------

            # Is this term a special term?
            if "ST" in list(modifier_log.keys()):

                # Yes, it is. Add additional modules to load, if necessary
                if "import scipy.special" not in modules:
                    modules     .append("import scipy.special")

                # Extract this special term's dimension
                idx = modifier_log["ST"]

                # Replace __mu__ with the correct ST location variable
                term = term.replace(
                    "__mu__",
                    "self.special_terms["+str(k+self.skip_dimensions)+"]["+str(idx)+"]['centers']["+str(ST_counter[idx])+"]")

                # Replace __scale__ with the correct ST location variable
                term = term.replace(
                    "__scale__",
                    "self.special_terms["+str(k+self.skip_dimensions)+"]["+str(idx)+"]['scales']["+str(ST_counter[idx])+"]")

                # Increment the special term counter
                ST_counter[idx] += 1

            # -------------------------------------------------------------
            # Add the term to the list
            # -------------------------------------------------------------

            # If any dummy __x__ remain, replace them
            term = term.replace("__x__", "x")

            # Store the term
            terms   .append(copy.copy(term))

        # =================================================================
        # Assemble the monotone function
        # =================================================================

        # Only assemble the function if there actually is a nonmonotone term
        if len(self.nonmonotone[k]) > 0:

            # Prepare the basis string
            string = "def fun(x,self):\n\t\n\t"

            # -------------------------------------------------------------
            # Load module requirements
            # -------------------------------------------------------------

            for entry in modules:
                string += copy.copy(entry)+"\n\t"
            string += "\n\t"  # Another line break for legibility

            # -------------------------------------------------------------
            # Prepare linearization, if necessary
            # -------------------------------------------------------------

            # If linearization is active, truncate the input x
            if self.linearization is not None:

                # First, find our which parts are outside the linearization hypercube
                string += "vec_below = copy.copy(x) - self.linearization_threshold[:,0][np.newaxis,:];\n\t"
                # Set all values above to zero
                string += "vec_below[vec_below >= 0] = 0;\n\t"
                string += "vec_above = copy.copy(x) - self.linearization_threshold[:,1][np.newaxis,:];\n\t"
                # Set all values below to zero
                string += "vec_above[vec_above <= 0] = 0;\n\t"
                string += "vec = vec_above + vec_below;\n\t"

                # Then convert the two arrays to boolean markers
                # Find all particles BELOW the lower linearization band
                string += "below = (vec_below < 0);\n\t"
                # Find all particles ABOVE the upper linearization band
                string += "above = (vec_above > 0);\n\t"
                # This is a matrix where all entries outside the linearization bands are 1 and all entries inside are 0
                string += "shift = np.asarray(below,dtype=float) + np.asarray(above,dtype=float);\n\t"

                # Truncate all values outside the hypercube
                string += "x_trc = copy.copy(x);\n\t"
                string += "for d in range(x.shape[1]):\n\t\t"
                # All values below the linearization band of this dimension are snapped to its border
                string += "x_trc[below[:,d],d] = self.linearization_threshold[d,0];\n\t\t"
                # All values above the linearization band of this dimension are snapped to its border
                string += "x_trc[above[:,d],d] = self.linearization_threshold[d,1];\n\t"

                # Add a space to the next block
                string += "\n\t"

                # Also crate an extrapolated version of x_trc
                string += "x_ext = copy.copy(x_trc);\n\t"
                # Offset all values which have been snapped by a small increment
                string += "x_ext += shift*" + \
                    str(self.linearization_increment)+";\n\t"

                # Add a space to the next block
                string += "\n\t"

            # -------------------------------------------------------------
            # Prepare precalculated variables
            # -------------------------------------------------------------

            # Add all precalculation terms
            for key in list(dict_precalc.keys()):

                string += key + " = " + copy.copy(dict_precalc[key]) + ";\n\t"

            # -------------------------------------------------------------
            # Assemble function output
            # -------------------------------------------------------------

            # Prepare the result string
            if len(terms) == 1:  # Only a single term, no need for stacking

                string += "result = " + \
                    copy.copy(terms[0])+"[:,np.newaxis];\n\t\n\t"

            else:  # If we have more than one term, start stacking the result

                # Prepare the stack
                string += "result = np.stack((\n\t\t"

                # Go through each entry in terms, add them one by one
                for entry in terms:
                    string += copy.copy(entry) + ",\n\t\t"

                # Remove the last ",\n\t\t" and close the stack
                string = string[:-4]
                string += "),axis=-1)\n\t\n\t"

            # Return the result
            string += "return result"

            # -------------------------------------------------------------
            # Finish function construction
            # -------------------------------------------------------------

            if not partial_construction:

                # Append the function string
                self.fun_nonmon_strings    .append(string)

                # Create an actual function
                funstring = "fun_nonmon_"+str(k)
                exec(string.replace("fun", funstring), globals())
                exec("self.fun_nonmon.append(copy.deepcopy("+funstring+"))")

            else:

                # Insert the function string
                self.fun_nonmon_strings[k] = copy.copy(string)

                # Create an actual function
                funstring = "fun_nonmon_"+str(k)
                exec(string.replace("fun", funstring), globals())
                exec("self.fun_nonmon[k] = copy.deepcopy("+funstring+")")

        else:  # There are NO non-monotone terms

            # Create a function which just returns None
            string = "def fun(x,self):\n\t"
            string += "return None"

            if not partial_construction:

                # Append the function string
                self.fun_nonmon_strings     .append(string)

                # Create an actual function
                funstring = "fun_nonmon_"+str(k)
                exec(string.replace("fun", funstring), globals())
                exec("self.fun_nonmon.append(copy.deepcopy("+funstring+"))")

            else:

                # Insert the function string
                self.fun_nonmon_strings[k] = copy.copy(string)

                # Create an actual function
                funstring = "fun_nonmon_"+str(k)
                exec(string.replace("fun", funstring), globals())
                exec("self.fun_nonmon[k] = copy.deepcopy("+funstring+")")

    # =================================================================
    # =================================================================
    # Step 3: Finalize
    # =================================================================
    # =================================================================

    # If monotonicity mode is 'separable monotonicity', we also require the
    # derivative of the monotone part of the map
    if self.monotonicity.lower() == "separable monotonicity":

        self.function_derivative_constructor_alternative()

    return

def function_derivative_constructor_alternative(self):
    """
    This function is the complement to 'function_constructor_alternative',
    but instead constructs the derivative of the map's component functions.
    It constructs the functions' strings, then converts them into
    functions.
    """

    import numpy as np
    import copy

    self.der_fun_mon = []
    self.der_fun_mon_strings = []

    self.optimization_constraints_lb = []
    self.optimization_constraints_ub = []

    # Find out how many function terms we are building
    K = len(self.monotone)

    # Go through all terms
    for k in range(K):

        # =================================================================
        # =================================================================
        # Step 1: Build the monotone function
        # =================================================================
        # =================================================================

        # Set optimization constraints
        self.optimization_constraints_lb.append(
            np.zeros(len(self.monotone[k])))
        self.optimization_constraints_ub.append(
            np.ones(len(self.monotone[k]))*np.inf)

        # Define modules to load
        modules = ["import numpy as np", "import copy"]

        # Define the terms composing the transport map component
        terms = []

        # Prepare a counter for the special terms
        ST_counter = np.zeros(self.X.shape[-1], dtype=int)

        # Mark which of these are special terms, in case we want to create
        # permutations of multiple RBFS
        ST_indices = []

        # Go through all terms, extract terms for precalculation
        dict_precalc = {}
        for j, entry in enumerate(self.monotone[k]):

            # -------------------------------------------------------------
            # Convert the map specification to a function
            # -------------------------------------------------------------

            # Find the term's function
            term, modifier_log = self.write_basis_function(
                term=entry,
                mode='derivative',
                k=k + self.skip_dimensions)

            # -------------------------------------------------------------
            # If this is a constant term, undo the lower constraint
            # -------------------------------------------------------------

            if "constant" in list(modifier_log.keys()):

                # Assign linear constraints
                self.optimization_constraints_lb[k][j] = -np.inf
                self.optimization_constraints_ub[k][j] = +np.inf

            # -------------------------------------------------------------
            # Extract any precalculations, where applicable
            # -------------------------------------------------------------

            # If this term includes and precalculations, extract them
            if "variables" in list(modifier_log.keys()):

                # There are precalculating variables. Go through each
                for key in list(modifier_log["variables"].keys()):

                    # Have we logged this one already?
                    if key not in list(dict_precalc.keys()):

                        # No, we haven't. Add it.
                        dict_precalc[key] = copy.copy(
                            modifier_log["variables"][key]).replace("__x__", "x")

            # -------------------------------------------------------------
            # Post-processing for special terms
            # -------------------------------------------------------------

            # Is this term a special term?
            if "ST" in list(modifier_log.keys()):

                # Mark this term as a special one
                ST_indices.append(j)

                # Yes, it is. Add additional modules to load, if necessary
                if "import scipy.special" not in modules:
                    modules     .append("import scipy.special")

                # Extract this special term's dimension
                idx = modifier_log["ST"]

                # Is this a cross-term?
                # Cross-terms are stored in a separate key; access it, if
                # necessary.
                if k+self.skip_dimensions != idx:
                    # Yes, it is.
                    ctkey = "['cross-terms']"
                else:
                    # No, it isn't.
                    ctkey = ""

                # Replace __mu__ with the correct ST location variable
                term = term.replace(
                    "__mu__",
                    "self.special_terms["+str(k+self.skip_dimensions)+"]"+ctkey+"["+str(idx)+"]['centers']["+str(ST_counter[idx])+"]")

                # Replace __scale__ with the correct ST location variable
                term = term.replace(
                    "__scale__",
                    "self.special_terms["+str(k+self.skip_dimensions)+"]"+ctkey+"["+str(idx)+"]['scales']["+str(ST_counter[idx])+"]")

                # Increment the special term counter
                ST_counter[idx] += 1

            # -------------------------------------------------------------
            # Add the term to the list
            # -------------------------------------------------------------

            # If any dummy __x__ remain, replace them
            term = term.replace("__x__", "x")

            # Store the term
            terms   .append(copy.copy(term))

        # Are there multiple special terms?
        # if np.sum([True if x != 0 else False for x in self.RBF_counter_m[k,:]]) > 1:
        # if np.sum([True if x != k else False for x in list(self.special_terms[k].keys())]) > 1:
        if 'cross-terms' in list(self.special_terms[k+self.skip_dimensions].keys()):

            import itertools

            # Yes, there are multiple special terms. Extract these terms.
            RBF_terms = [terms[i] for i in ST_indices]

            # Check what variables these terms are affiliated with
            RBF_terms_dim = - np.ones(len(RBF_terms), dtype=int)
            for ki in range(k+1+self.skip_dimensions):
                for i, term in enumerate(RBF_terms):
                    if "x[...,"+str(ki)+"]" in term:
                        RBF_terms_dim[i] = ki
            RBF_terms_dims = np.unique(np.asarray(RBF_terms_dim))

            # Create a dictionary with the different terms
            RBF_terms_dict = {}
            for i in RBF_terms_dims:
                RBF_terms_dict[i] = [RBF_terms[j] for j in range(
                    len(RBF_terms)) if RBF_terms_dim[j] == i]

            # Create all combinations of terms
            RBF_terms_grid = copy.deepcopy(RBF_terms_dict[RBF_terms_dims[0]])
            for i in RBF_terms_dims[1:]:

                # Create a grid with the next dimension
                RBF_terms_grid = list(itertools.product(
                    RBF_terms_grid,
                    copy.deepcopy(RBF_terms_dict[i])))

                # Convert this list of tuples into a new list of strings
                RBF_terms_grid = \
                    [entry[0]+"*"+entry[1] for entry in RBF_terms_grid]

            # Now remove all original RBF terms
            terms = [entry for i, entry in enumerate(
                terms) if i not in ST_indices]

            # Now add all the grid terms
            terms += RBF_terms_grid

        # =================================================================
        # Assemble the monotone derivative function
        # =================================================================

        # Prepare the basis string
        string = "def fun(x,self):\n\t\n\t"

        # -----------------------------------------------------------------
        # Load module requirements
        # -----------------------------------------------------------------

        # Add all module requirements
        for entry in modules:
            string += copy.copy(entry)+"\n\t"
        string += "\n\t"  # Another line break for legibility

        # -----------------------------------------------------------------
        # Prepare linearization, if necessary
        # -----------------------------------------------------------------

        # If linearization is active, truncate the input x
        if self.linearization is not None:

            # First, find our which parts are outside the linearization hypercube
            string += "vec_below = self.linearization_threshold[:,0][np.newaxis,:] - x;\n\t"
            # Set all values above to zero
            string += "vec_below[vec_below > 0] = 0;\n\t"
            string += "vec_above = x - self.linearization_threshold[:,1][np.newaxis,:];\n\t"
            # Set all values below to zero
            string += "vec_above[vec_above > 0] = 0;\n\t"
            string += "vec = vec_above + vec_below;\n\t"

            # Then convert the two arrays to boolean markers
            string += "below = (vec_below < 0)\n\t"
            string += "above = (vec_above > 0);\n\t"
            string += "vecnorm = np.asarray(below,dtype=float) + np.asarray(above,dtype=float);\n\t"

            # Truncate all values outside the hypercube
            string += "for d in range(x.shape[1]):\n\t\t"
            string += "x[below[:,d],d] = self.linearization_threshold[d,0];\n\t\t"
            string += "x[above[:,d],d] = self.linearization_threshold[d,1];\n\t"

            # Add a space to the next block
            string += "\n\t"

            # The derivative of a linearized function outside its range is
            # constant, so we do not require x_ext

        # -----------------------------------------------------------------
        # Prepare precalculated variables
        # -----------------------------------------------------------------

        # Add all precalculation terms
        for key in list(dict_precalc.keys()):

            string += key + " = " + copy.copy(dict_precalc[key]) + ";\n\t"

        # -----------------------------------------------------------------
        # Assemble function output
        # -----------------------------------------------------------------

        # Prepare the result string
        if len(terms) == 1:  # Only a single term, no need for stacking

            string += "result = "+copy.copy(terms[0])+"[:,np.newaxis];\n\t\n\t"

        else:  # If we have more than one term, start stacking the result

            # Prepare the stack
            string += "result = np.stack((\n\t\t"

            # Go through each entry in terms, add them one by one
            for entry in terms:
                string += copy.copy(entry) + ",\n\t\t"

            # Remove the last ",\n\t\t" and close the stack
            string = string[:-4]
            string += "),axis=-1)\n\t\n\t"

        # Return the result
        string += "return result"

        # -----------------------------------------------------------------
        # Finish function construction
        # -----------------------------------------------------------------

        # Append the function string
        self.der_fun_mon_strings    .append(string)

        # Create an actual function
        funstring = "der_fun_mon_"+str(k)
        exec(string.replace("fun", funstring), globals())
        exec("self.der_fun_mon.append(copy.deepcopy("+funstring+"))")

    return

if __name__ == "__main__":
    term = [0,0,1,1]
    polyfunc = np.polynomial.HermiteE
    polyfunc_str = "np.polynomial.hermite_e.hermeval"
    polyfunc_der = None
    write_polynomial_basis_function(term, polyfunc, polyfunc_str, polyfunc_der)