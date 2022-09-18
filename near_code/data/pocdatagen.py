import numpy as np
import argparse

def dsl_function1(inp):
    x = inp
    if (x < -0.5):
        return 0
    if (2.45 * x < 1):
        return 1
    if (x > 1.5):
        return 2
    return 3

def dsl_function2(inp):
    x, y = inp[0], inp[1]
    if (0.4*x - 3*y < -0.5):
        return 0
    if (2.45 * x < 1):
        return 1
    if (x + y*2.5 > 1.5):
        return 2
    if (3*x - 23*y > 0.5):
        return 3
    return 4

def dsl_function3(inp):
    x, y = inp[0], inp[1]
    if (x+y < -0.5):
        if (0.4*x - 3*y < -0.5):
            if (3*x - 23*y > 0.5):
                return 0
            elif (x + y*2.5 > 1.5):
                return 1
            else:
                return 2
        else:
            return 3
    else:
        if (2.45 * x < 1):
            return 4
        elif (x > 1.5):
            return 5
        else:
            return 6

size = 40
def generate_condition():
    return np.random.randn(size)
conditions_store = [generate_condition() for _ in range(15)]
def dsl_function4(inp):
    def apply_condition(c):
        return np.sum(c * inp) + 1 > 0
    if (apply_condition(conditions_store[0])):
        if (apply_condition(conditions_store[1])):
            if (apply_condition(conditions_store[2])):
                return 0
            elif (apply_condition(conditions_store[3])):
                return 1
            else:
                return 2
        elif (apply_condition(conditions_store[4])):
            return 3
        else:
            return 4
    else:
        if (apply_condition(conditions_store[5])):
            return 5
        elif (apply_condition(conditions_store[6])):
            return 6
        else:
            return 7

def dsl_function5(inp):
    def apply_condition(c):
        return np.sum(c * inp) + 1 > 0
    if (apply_condition(conditions_store[0])):
        if (apply_condition(conditions_store[1])):
            if (apply_condition(conditions_store[2])):
                return 0
            else:
                return 1
        else:
            return 2
    else:
        if (apply_condition(conditions_store[5])):
            if (apply_condition(conditions_store[6])):
                return 3
            else:
                return 4
        else:
            return 5


dsl_functions = [dsl_function1, dsl_function2, dsl_function3, dsl_function4, dsl_function5]
func_inp_map = dict(zip(dsl_functions, [1,2,2,40,40]))
dsl_function_vecs = [np.vectorize(func) for func in func_inp_map.keys()]

def generate_data(dsl_func_key, num_steps_train, prefix, key='train'):
    X = np.random.randn(num_steps_train,func_inp_map[dsl_functions[dsl_func_key]]) * 2 + 1
    # map X with dsl_function
    Y = np.array([dsl_functions[dsl_func_key](x) for x in X])[:,None]
    # save data
    print(f"Saving {X.shape} and {Y.shape}")
    np.save(prefix + key + '_X.npy', X)
    np.save(prefix + key + '_Y.npy', Y)
    return

if __name__ == '__main__':
    # generate data
    def Opts():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dsl_func_key', type=int, default=4)        
        parser.add_argument('--num_steps_train', type=int, default=500)
        parser.add_argument('--num_steps_test', type=int, default=200)
        parser.add_argument('--prefix', type=str, default='data/')
        return parser.parse_args()
    opts = Opts()
    generate_data(opts.dsl_func_key, opts.num_steps_train, opts.prefix, key='train')
    generate_data(opts.dsl_func_key, opts.num_steps_test, opts.prefix, key='test')
