import numpy as np
from functools import partial
import inspect


def cosine_decay(initial_value, min_value, epoch, total_epochs, stay_at_min=True):
    """
    Cosine decay function.

    Args:
        initial_value (float): The starting value.
        min_value (float): The minimum value to decay to.
        epoch (int): The current epoch number.
        total_epochs (int): The total number of epochs over which to decay.
        stay_at_min (bool): If True, the value remains at min_value after reaching it.

    Returns:
        float: The decayed value.
    """
    if epoch > total_epochs and stay_at_min:
        return min_value
    cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
    decayed_value = (initial_value - min_value) * cosine_decay + min_value
    return decayed_value

def cosine_decay_reversed(min_value, initial_value, epoch, total_epochs, stay_at_max=True):
    if epoch > total_epochs and stay_at_max:
        return initial_value
    cosine_growth = 0.5 * (1 + np.cos(np.pi * (1 - epoch / total_epochs)))
    grown_value = (initial_value - min_value) * cosine_growth + min_value
    return grown_value

def linear_decay(initial_value, min_value, epoch, total_epochs, stay_at_min=True):
    """
    Linear decay function.

    Args:
        initial_value (float): The starting value.
        min_value (float): The minimum value to decay to.
        epoch (int): The current epoch number.
        total_epochs (int): The total number of epochs over which to decay.
        stay_at_min (bool): If True, the value remains at min_value after reaching it.

    Returns:
        float: The decayed value.
    """
    if epoch > total_epochs and stay_at_min:
        return min_value
    linear_decay = 1 - (epoch / total_epochs)
    decayed_value = (initial_value - min_value) * linear_decay + min_value
    return decayed_value

def linear_decay_reversed(min_value, initial_value, epoch, total_epochs, stay_at_max=True):
    if epoch > total_epochs and stay_at_max:
        return initial_value
    linear_growth = epoch / total_epochs
    grown_value = (initial_value - min_value) * linear_growth + min_value
    return grown_value

def exponential_decay(initial_value, min_value, epoch, decay_rate, power=2, stay_at_min=True):
    """
    Steeper Exponential decay function.

    Args:
        initial_value (float): The starting value.
        min_value (float): The minimum value to decay to.
        epoch (int): The current epoch number.
        decay_rate (float): The decay rate (must be between 0 and 1).
        power (float): Exponent to make the curve steeper.
        stay_at_min (bool): If True, the value remains at min_value after reaching it.

    Returns:
        float: The decayed value.
    """
    effective_decay = decay_rate ** (epoch ** power)
    if stay_at_min:
        decayed_value = max(min_value, initial_value * effective_decay)
    else:
        decayed_value = initial_value * effective_decay
    return decayed_value

def inverted_exponential_growth(min_value, max_value, epoch, growth_rate, power=2, converge_to_max=True):
    effective_growth = 1 - (growth_rate ** (epoch ** power))
    value = min_value + (max_value - min_value) * effective_growth
    if converge_to_max:
        return min(max_value, value)
    else:
        return value


def exponential_decay_reversed(min_value, initial_value, epoch, growth_rate, stay_at_max=True):
    if stay_at_max:
        grown_value = min(initial_value, min_value + (initial_value - min_value) * (1 - growth_rate ** epoch))
    else:
        grown_value = min_value + (initial_value - min_value) * (1 - growth_rate ** epoch)
    return grown_value

def stepwise_decay(initial_value, min_value, epoch, decay_steps, decay_amount, stay_at_min=True):
    """
    Stepwise decay function.

    Args:
        initial_value (float): The starting value.
        min_value (float): The minimum value to decay to.
        epoch (int): The current epoch number.
        decay_steps (int): Number of epochs between each decay step.
        decay_rate (float): Factor by which the value decreases at each step.
        stay_at_min (bool): If True, the value remains at min_value after reaching it.

    Returns:
        float: The decayed value.
    """
    num_steps = epoch // decay_steps 
    decayed_value = initial_value - num_steps * decay_amount 
    if stay_at_min:
        return max(min_value, decayed_value) 
    return decayed_value

def stepwise_decay_reversed(min_value, initial_value, epoch, growth_steps, growth_amount, stay_at_max=True):
    num_steps = epoch // growth_steps
    grown_value = min_value + num_steps * growth_amount
    if stay_at_max:
        return min(initial_value, grown_value)
    return grown_value

decays = {
    "cosine": cosine_decay,
    "linear": linear_decay,
    "exponential": exponential_decay,
    "stepwise": stepwise_decay,
}

def get_qmin_func(decay_type, safety_p, max_steps,  **kwargs):
    """
    Partially apply parameters to the chosen decay function.

    :param decay_type: The type of decay function as a string.
    :param kwargs: Parameters to partially apply to the function.
    :return: A partially-applied function pointer.
    """
    decay_func = None
    if decay_type is not None:
        if decay_type not in decays:
            raise ValueError(f"Unknown decay type: {decay_type}")
        decay_func = decays[decay_type]
        # Get the valid parameters for the function
        valid_params = inspect.signature(decay_func).parameters
        print(f"VALID PARAMS {valid_params}")
        # Filter out extra arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        # Warn about ignored arguments
        ignored_params = set(kwargs) - set(filtered_kwargs)
        if ignored_params:
            print(f"Warning: Ignoring unexpected parameters: {ignored_params}")
        decay_func = partial(decays[decay_type], **filtered_kwargs)
        
    def get_q_min_thresh(current_epoch):
        if decay_func is None:
            return -1 * (1. - safety_p) * max_steps, safety_p
        safety_param = 1 - decay_func(epoch=current_epoch)
        return -1 * (1. - safety_param) * max_steps, safety_param
    
    return get_q_min_thresh

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    initial_value = 0.9
    min_value = 0.6
    total_epochs = 1e6
    decay_rate = 0.9999999999995
    epochs = np.arange(0, total_epochs) 
    import matplotlib.pyplot as plt
    
    
    cosine_values = [1 - cosine_decay(initial_value, min_value, epoch, total_epochs) for epoch in epochs]
    linear_values = [1 - linear_decay(initial_value, min_value, epoch, total_epochs) for epoch in epochs]
    exponential_values = [1 - exponential_decay(initial_value, min_value, epoch, decay_rate) for epoch in epochs]
    step_values = [1 - stepwise_decay(initial_value, min_value, epoch, decay_steps=total_epochs/10, decay_amount=0.1) for epoch in epochs]
    

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, cosine_values, label='Cosine Decay')
    plt.plot(epochs, linear_values, label='Linear Decay')
    plt.plot(epochs, exponential_values, label='Exponential Decay')
    plt.plot(epochs, step_values, label='Step Decay')
    plt.axhline(y=1-min_value, color='r', linestyle='--', label='Minimum Value')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Decay Schedules')
    plt.legend()
    plt.grid(True)
    plt.savefig("decays.png")