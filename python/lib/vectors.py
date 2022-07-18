
import math 

def unit_direction(x):
    """
    Compute nuit vector in direction of x 

    Returns: y - unit vector whch direction is same as of x 
    """
    return x / math.sqrt(x.dot(x))


