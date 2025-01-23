import ctypes
import numpy as np
import os
import platform
from numpy.ctypeslib import ndpointer

# Load the compiled C library with proper path handling
try:
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine library extension based on OS
    if platform.system() == 'Darwin':
        lib_name = 'libgrpo.dylib'
    elif platform.system() == 'Linux':
        lib_name = 'libgrpo.so'
    else:
        raise OSError("Unsupported operating system")
    
    # Construct full path to library
    lib_path = os.path.join(current_dir, 'c_src', lib_name)
    
    # Load the library
    lib = ctypes.CDLL(lib_path)
except Exception as e:
    raise RuntimeError(f"Failed to load C library: {str(e)}") from e

# Define the GRPOBatch structure
class GRPOBatch(ctypes.Structure):
    _fields_ = [
        ('log_probs_old', ctypes.POINTER(ctypes.c_double)),
        ('log_probs_ref', ctypes.POINTER(ctypes.c_double)),
        ('rewards', ctypes.POINTER(ctypes.c_double)),
        ('group_size', ctypes.c_int)
    ]

# Define function prototypes with proper restype declarations
lib.compute_advantages.argtypes = [
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # rewards
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # advantages
    ctypes.c_int                                       # group_size
]
lib.compute_advantages.restype = None

lib.grpo_loss.argtypes = [
    ctypes.POINTER(GRPOBatch),     # batch
    ndpointer(ctypes.c_double),    # log_probs_new
    ctypes.POINTER(ctypes.c_double),  # loss
    ndpointer(ctypes.c_double),    # grad
    ctypes.c_double,               # epsilon
    ctypes.c_double                # beta
]
lib.grpo_loss.restype = None

class GRPO:
    def __init__(self, epsilon=0.2, beta=0.1):
        """
        Initialize GRPO optimizer
        
        :param epsilon: Clipping range parameter (default: 0.2)
        :param beta: KL penalty coefficient (default: 0.1)
        """
        self.epsilon = epsilon
        self.beta = beta

    def compute_loss(self, batch_data, log_probs_new):
        """
        Compute GRPO loss and gradients
        
        :param batch_data: Dictionary containing:
            - log_probs_old: Array of old policy log probabilities
            - log_probs_ref: Array of reference policy log probabilities
            - rewards: Array of rewards
            - group_size: Integer size of the group
        :param log_probs_new: Array of new policy log probabilities
        :return: Tuple of (loss value, gradients array)
        """

            # Validate input types
        for key in ['log_probs_old', 'log_probs_ref', 'rewards']:
            if batch_data[key].dtype != np.float64:
                raise ValueError(f"{key} must be float64 (np.float64)")
                
        if log_probs_new.dtype != np.float64:
            raise ValueError("log_probs_new must be float64 (np.float64)")

        # Validate input shapes
        if len(log_probs_new) != batch_data['group_size']:
            raise ValueError("log_probs_new length must match group_size")
        # # Validate input shapes
        # if len(log_probs_new) != batch_data['group_size']:
        #     raise ValueError("log_probs_new length must match group_size")
            
        # Prepare batch structure
        batch = GRPOBatch(
            batch_data['log_probs_old'].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            batch_data['log_probs_ref'].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            batch_data['rewards'].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(batch_data['group_size'])
        )

        loss = ctypes.c_double(0.0)
        grad = np.zeros_like(log_probs_new, dtype=np.float64)

        # Call the C function
        lib.grpo_loss(
            ctypes.byref(batch),
            log_probs_new.astype(np.float64),
            ctypes.byref(loss),
            grad,
            ctypes.c_double(self.epsilon),
            ctypes.c_double(self.beta)
        )
        
        return loss.value, grad

if __name__ == '__main__':
    # Example usage
    group_size = 5
    batch_data = {
        'log_probs_old': np.random.randn(group_size).astype(np.float64),
        'log_probs_ref': np.random.randn(group_size).astype(np.float64),
        'rewards': np.random.randn(group_size).astype(np.float64),
        'group_size': group_size
    }
    log_probs_new = np.random.randn(group_size).astype(np.float64)

    grpo = GRPO(epsilon=0.2, beta=0.1)
    loss, grad = grpo.compute_loss(batch_data, log_probs_new)
    print(f"Loss: {loss}\nGradients: {grad}")