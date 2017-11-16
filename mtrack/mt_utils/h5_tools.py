import h5py
import os

def stack_to_h5(input_stack, output_stack):
    if not os.path.exists(os.path.dirname(output_stack)):
        os.makedirs(os.path.dirname(output_stack))

    f = h5py.File(output_stack, 'w')
    f.create_dataset('exported_data', data=input_stack)
    f.close()
