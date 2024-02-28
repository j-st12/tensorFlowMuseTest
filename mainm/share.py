import multiprocessing
import numpy as np
from sklearn.utils import shuffle
import os

def save_on_shared_array(pipe, data_dir, use_only_84_keys=True, rescale=True, postfix=''):
    print('Reading...')
    print('[*]', data_dir)

    subdirs = ['tra', 'val']

    for sd in subdirs:
        data_X = np.load(r'\Users\jls1\PycharmProjects\muse\saved_roll\x_bar_chroma_Amazing_Grace.npy')
        if sd == 'tra':
            data_y = np.load(r'C:\Users\jls1\PycharmProjects\muse\saved_roll\y_bar_chroma_Amazing_Grace.npy')
            print(sd)
            print('Shuffling...')
            data_X, data_y = shuffle(data_X, data_y, random_state=0)
        else:
            print(sd)
            pass

        name_x = sd + '_X_' + postfix
        shared_array_x = multiprocessing.Array('d', data_X.flatten())
        np.copyto(np.frombuffer(shared_array_x.get_obj()).reshape(data_X.shape), data_X)

        name_y = sd + '_y_' + postfix
        shared_array_y = multiprocessing.Array('d', data_y.flatten())
        np.copyto(np.frombuffer(shared_array_y.get_obj()).reshape(data_y.shape), data_y)

        # Send the shared arrays through the pipe
        pipe.send((name_x, shared_array_x, name_y, shared_array_y))

if __name__ == '__main__':
    parent_pipe, child_pipe = multiprocessing.Pipe()
    data_dir = '/content/data'
    postfix = 'bars'

    process = multiprocessing.Process(target=save_on_shared_array, args=(child_pipe, data_dir, postfix))
    process.start()

    # Retrieve shared arrays from the pipe
    name_x, shared_array_x, name_y, shared_array_y = parent_pipe.recv()

    # Now you can use shared_array_x and shared_array_y as needed

    process.join()