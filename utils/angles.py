import math

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points:
    a, b, c are (x, y) tuples.
    Angle is measured at point b.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)

    ## AB⋅CB=∣AB∣⋅∣CB∣⋅cos(θ)
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_cb = math.sqrt(cb[0]**2 + cb[1]**2)

    if mag_ab == 0 or mag_cb == 0:
        return 0

    cos_angle = dot / (mag_ab * mag_cb)
    cos_angle = max(min(cos_angle, 1), -1)  # clamp to avoid errors
    
    return math.degrees(math.acos(cos_angle))
