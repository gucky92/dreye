"""
Get Intensities from contrast values
or vice versa
"""

import numpy as np


def ints_from_prop_samples(
    combo, # 2d-array of which leds to combine
    samples, 
    bg_ints, 
    equal='contrast'
):
    assert combo.shape[0] == samples.shape[1]
    assert combo.shape[1] == bg_ints.size
    d = combo.shape[0]

    if equal == 'contrast':
        # total contrast of zero
        contrasts = (samples - 1/d) * d  # contrast changes for each channel -> from -1 to D
        contrasts = (
            combo 
            / combo.sum(1, keepdims=True) 
            * contrasts[..., None]
        ).sum(1)
        return bg_ints + contrasts * bg_ints
    elif equal == 'intensity':
        # equal intensity
        # same intensity different contrasts not centered
        return (
            samples[..., None] 
            * combo 
            / combo.sum(1, keepdims=True) 
            * bg_ints.sum()
        ).sum(axis=1)
    elif equal is None:
        return (
            samples[..., None] 
            * combo 
            * bg_ints 
            * d
        ).sum(axis=1)
    else:
        raise NameError(f"`equal` cannot be `{equal}`.")


def absolutes_from_michelson_contrast(
    contrasts, 
    bg_ints, 
    led_idx1, 
    led_idx2
):
    bg1 = bg_ints[led_idx1].sum()
    bg2 = bg_ints[led_idx2].sum()
    denom = bg1 + bg2
    change_in_led1 = (contrasts * denom - bg1 + bg2)/2

    abs_in_led1 = change_in_led1 + bg1
    abs_in_led2 = -change_in_led1 + bg2

    led_ints = np.broadcast_to(
        bg_ints, 
        (contrasts.size, bg_ints.size)
    ).copy()

    led_ints[:, led_idx1] = bg_ints[led_idx1] / bg1 * abs_in_led1[:, None]
    led_ints[:, led_idx2] = bg_ints[led_idx2] / bg2 * abs_in_led2[:, None]

    return led_ints


def michelson_contrast_from_absolutes(
    ints,
    led_idx1, 
    led_idx2
):
    num = ints[:, led_idx1].sum(1) - ints[:, led_idx2].sum(1)
    denom = ints[:, led_idx1].sum(1) + ints[:, led_idx2].sum(1)
    return num / denom