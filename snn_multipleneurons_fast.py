import numpy as np
import time
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix, identity

# if you want to use cupy for gpu 
# import cupy as cp
# from cupyx.scipy.sparse import csr_matrix as csr_gpu

# if you just wnat to use numpy and scipy for cpu
import numpy as cp
from scipy.sparse import csr_matrix as csr_gpu

class network_weights(object):
    def __init__(self, NpS, previous_NpS, distance_parameter, input_dim, stride, lateral_distance):
        self.distance_parameter = distance_parameter
        self.input_dim = input_dim
        self.stride = stride
        self.NpS = NpS
        self.output_dim = int(self.input_dim/self.stride)
        self.W = None
        self.L = None
        self.W_structure = None
        self.L_structure = None
        self.previous_NpS = previous_NpS
        self.lateral_distance = lateral_distance
        
    def create_h_distances(self):
        distances = np.zeros((self.output_dim**2, self.input_dim, self.input_dim))
        dict_input_2_position = {}
        for row_index in range(self.input_dim):
            for column_index in range(self.input_dim):
                input_index = row_index*self.input_dim + column_index
                dict_input_2_position[row_index, column_index] = input_index
                
        centers = []
        dict_output_2_position = {}
        for i in range(self.output_dim):
            for j in range(self.output_dim):
                stride_padding = self.stride/2
                neuron_center = np.array([i*self.stride + stride_padding, j*self.stride + stride_padding])
                centers.append(neuron_center)
                neuron_index = i*self.output_dim + j
                dict_output_2_position[neuron_index] = neuron_center
                for k in range(self.input_dim):
                    for l in range(self.input_dim):
                        distances[neuron_index, k,l] = np.linalg.norm(np.array([k+0.5,l+0.5])-neuron_center)
        above_threshold = distances > self.distance_parameter
        below_threshold = distances <= self.distance_parameter
        distances[above_threshold] = 0
        distances[below_threshold] = 1
        distances = distances.reshape((self.output_dim**2, self.input_dim**2))
        return distances
    
    def create_ah_distances(self):
        centers = []
        dict_output_2_position = {}
        for i in range(self.output_dim):
            for j in range(self.output_dim):
                stride_padding = self.stride/2
                neuron_center = np.array([i*self.stride + stride_padding, j*self.stride + stride_padding])
                centers.append(neuron_center)
                neuron_index = i*self.output_dim + j
                dict_output_2_position[neuron_index] = neuron_center
        distances_ah = np.zeros((self.output_dim**2, self.output_dim**2))
        for row_index in list(dict_output_2_position.keys()):
            center = dict_output_2_position[row_index]
            for column_index in list(dict_output_2_position.keys()):
                other_center = dict_output_2_position[column_index]
                distances_ah[row_index, column_index] = np.linalg.norm(other_center - center)
        above_threshold = distances_ah > self.lateral_distance #*self.anti_hebbian_binary
        below_threshold = distances_ah <= self.lateral_distance #*self.anti_hebbian_binary
        distances_ah[above_threshold] = 0
        distances_ah[below_threshold] = 1
        return distances_ah
    
    def create_L(self):
        mat = self.create_ah_distances()
        blocks = [[mat]*self.NpS]*self.NpS
        L_mat = np.block(blocks)
        return L_mat
    
    def create_W(self):
        mat = self.create_h_distances()
        blocks = [[mat]*self.previous_NpS]*self.NpS
        W_mat = np.block(blocks)
        return W_mat
    
    def create_weights_matrix(self):
        self.W_structure = self.create_W()
        self.L_structure = self.create_L()
        factor = np.sqrt(((np.sum(self.W_structure)/self.NpS)/self.output_dim**2))
        self.W = self.W_structure*np.random.normal(0, 1, (self.W_structure.shape))/factor
        self.L = self.L_structure*np.identity(self.NpS * self.output_dim**2)


class training(network_weights):
    def __init__(self, NpS, input_dim, stride, distance_parameter, lr, decay, lr_floor, lateral_distance, channels):
        network_weights.__init__(self, NpS, channels, distance_parameter, input_dim, stride, lateral_distance)
        self.alpha = None
        self.alpha_term = None
        self.lr = lr
        self.forget_rate = 0.99
        self.tickers = []
        self.epoch = 0
        self.decay = decay
        self.lr_floor = lr_floor
        self.current_lr = None
        self.C_mat = None
        self.W_mat = None
        self.L_mat = None
        self.regulariser_term = None
        self.costs = []
        network_weights.create_weights_matrix(self)
        training.create_sparse_matrices(self)
    
    def create_sparse_matrices(self): 
        self.W = csr_matrix(self.W)
        self.L = csr_matrix(self.L)
        self.W_structure = csr_matrix(self.W_structure)
        self.L_structure = csr_matrix(self.L_structure)

    def create_cost_matrices(self):
        self.C_mat = np.zeros((self.input_dim**2, self.input_dim**2))
        self.W_mat = np.zeros((self.W.shape))
        self.L_mat = np.zeros((self.L.shape))
        self.regulariser_term = 0

    def activation_function(self, vec):
        return np.tanh(vec)

    def neural_dynamics(self, net_input, verbose = False, check_dynamics = False):
        u, r = np.zeros(self.NpS*self.output_dim**2,), np.zeros(self.NpS*self.output_dim**2,)
        delta = np.inf
        conversion_ticker = 0
        updates = 0
        L_mat= self.L - identity(self.L.shape[0])
        W_mat = self.W.dot(net_input)
        while updates < 2000:
            if delta < 1e-4:
                conversion_ticker = 1
                break
            lr = max(0.4/(1+0.01*updates), 0.1)
            delta_u = -u + W_mat - L_mat.dot(r)
            u += lr*delta_u
            r = self.activation_function(u)
            delta = np.linalg.norm(delta_u)/np.linalg.norm(u)
            updates += 1
        if verbose==True:
            return r, u
        else:
            return r, conversion_ticker

    def update_weights(self, net_output, net_input, n_images):
        self.current_lr = max(self.lr * (1/(1 + self.decay*self.epoch)), self.lr_floor)
        W_matrix = np.outer(net_output, net_input)
        L_matrix = np.outer(net_output, net_output)
        self.C_mat = (1-1/n_images)*self.C_mat + (1/n_images)*np.outer(net_input, net_input)
        self.W_mat = (1-1/n_images)*self.W_mat + (1/n_images)*self.W_structure.multiply(W_matrix)
        self.L_mat = (1-1/n_images)*self.L_mat + (1/n_images)*self.L_structure.multiply(L_matrix)
        self.regulariser_term = (1-1/n_images)*self.regulariser_term + (1/n_images)*training.return_integral(self, net_output)
        delta_W = W_matrix - self.W
        delta_L = 0.5*(L_matrix - self.L)
        self.W += self.current_lr*self.W_structure.multiply(delta_W)
        self.L += self.current_lr*self.L_structure.multiply(delta_L)

    def return_integral(self, r):
        term_1 = np.multiply(0.5,np.log(1-np.power(r, 2)))
        term_2 = np.multiply(r, np.arctanh(r))
        term_3 = np.multiply(0.5, np.power(r, 2))
        return np.sum(term_1+term_2-term_3)
  
    def train_network(self, epochs, images, verbose = False, check_dynamics = False):
        n_images = images.shape[0]
        for epoch in range(epochs):
            img_array = shuffle(images, random_state = epoch)
            training.create_cost_matrices(self)
            epoch_start = time.time()
            sum_tickers = 0
            for i, img in enumerate(img_array):
                net_input = img.flatten()
                net_output, conversion_ticker = training.neural_dynamics(self, net_input)
                sum_tickers += conversion_ticker
                training.update_weights(self, net_output, net_input, n_images)
            sum_costs = np.trace(self.C_mat.T @ self.C_mat) - 2*np.trace(self.W_mat.T @ self.W_mat) + np.trace(self.L_mat.T @ self.L_mat) + self.regulariser_term
            self.costs.append(sum_costs)
            epoch_end = time.time()
            if verbose == True:
                time_taken = epoch_end - epoch_start
                print('Epoch: {}\nLoss: {}\nConvergence: {}\nTime Taken: {}\nCurrent Learning Rate: {}\n\n'.format(self.epoch+1, self.costs[-1], sum_tickers/n_images, time_taken, self.current_lr))
            self.tickers.append(sum_tickers/n_images)
            self.epoch += 1


class deep_network(object):
    def __init__(self, image_dim, channels, NpSs, strides, distances, 
                 layers, gamma, lr, lr_floor, decay, distances_lateral, tanh_factors, mult_factors, euler_step):
        self.image_dim = image_dim
        self.channels = channels
        self.NpSs = NpSs
        self.strides = strides
        self.distances = distances
        self.lateral_distances = distances_lateral
        self.layers = layers
        self.gamma = gamma
        self.lr = lr
        self.lr_floor = lr_floor
        self.current_lr = None
        self.decay = decay
        self.conversion_tickers = []
        self.costs = []
        self.epoch=0
        self.deep_matrix_weights = None
        self.deep_matrix_structure = None
        self.deep_matrix_identity = None
        self.weights_adjustment_matrix = None
        self.weights_update_matrix = None
        self.grad_matrix = None
        self.n_images = None
        self.dict_weights = {}
        self.dimensions = []
        self.g_vec = None
        self.mult_vec = None
        self.euler_step = euler_step
        self.tanh_factors = tanh_factors
        self.mult_factors = mult_factors
        deep_network.create_deep_network(self)
        deep_network.create_g_vec(self)
        deep_network.create_mult_vec(self)
        #deep_network.rescale_activation(self)
    
    def create_deep_network(self):
        for i in range(self.layers+1):
            dim = int(np.prod(self.strides[:i]))
            self.dimensions.append(int((self.image_dim/dim)**2)*([self.channels]+self.NpSs)[i])
        
        for i in range(self.layers):
            layer_input_dim = int(self.image_dim/np.prod(self.strides[:i]))
            self.dict_weights[i]=network_weights(NpS=([self.channels]+self.NpSs)[i+1], distance_parameter=self.distances[i], 
                                                input_dim=layer_input_dim,
                                                stride = self.strides[i], previous_NpS = ([self.channels]+self.NpSs)[i], lateral_distance=self.lateral_distances[i])
            self.dict_weights[i].create_weights_matrix()
        
        matrix_block = []
        structure_block = []
        matrix_identity = []
        weight_adjustment_block = []
        gradient_update_block = []

        for i, ele_row in enumerate(self.dimensions):
            row_block = []
            struc_block = []
            row_identity_block = []
            weights_adj_block = []
            grad_update_block = []

            start_block = max(i-1, 0)
            end_block = max(len(self.dimensions)-start_block-3, 0)

            if i == 0:
                row_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                struc_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                row_identity_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                weights_adj_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                grad_update_block.append(np.zeros((ele_row, np.sum(self.dimensions))))

            elif i < len(self.dimensions)-1:
                if start_block > 0:
                    row_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    row_identity_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    weights_adj_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    grad_update_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))

                row_block.append(self.dict_weights[i-1].W)
                row_block.append(self.dict_weights[i-1].L)
                row_block.append(self.dict_weights[i].W.T)
                
                struc_block.append(self.dict_weights[i-1].W_structure/self.mult_factors[i-1])
                struc_block.append(-self.dict_weights[i-1].L_structure)
                struc_block.append(self.gamma*self.mult_factors[i]*self.dict_weights[i].W_structure.T)
                
                row_identity_block.append(np.zeros((self.dict_weights[i-1].W_structure.shape)))
                row_identity_block.append(np.identity(self.dict_weights[i-1].L_structure.shape[0]))
                row_identity_block.append(np.zeros((self.dict_weights[i].W_structure.T.shape)))

                weights_adj_block.append(self.dict_weights[i-1].W_structure)
                weights_adj_block.append(self.dict_weights[i-1].L_structure/(1+self.gamma))
                weights_adj_block.append(self.dict_weights[i].W_structure.T)

                grad_update_block.append(self.dict_weights[i-1].W_structure)
                grad_update_block.append(self.dict_weights[i-1].L_structure/2)
                grad_update_block.append(self.dict_weights[i].W_structure.T)

                if end_block > 0:
                    row_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    row_identity_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    weights_adj_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    grad_update_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))

            elif i+1 == len(self.dimensions):
                if start_block > 0:
                    row_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    row_identity_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    weights_adj_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    grad_update_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))

                row_block.append(self.dict_weights[i-1].W)
                row_block.append(self.dict_weights[i-1].L)
                
                struc_block.append(self.dict_weights[i-1].W_structure/self.mult_factors[i-1])
                struc_block.append(-self.dict_weights[i-1].L_structure)
                
                row_identity_block.append(np.zeros((self.dict_weights[i-1].W_structure.shape)))
                row_identity_block.append(np.identity(self.dict_weights[i-1].L_structure.shape[0]))
                
                weights_adj_block.append(self.dict_weights[i-1].W_structure)
                weights_adj_block.append(self.dict_weights[i-1].L_structure)
                
                grad_update_block.append(self.dict_weights[i-1].W_structure)
                grad_update_block.append(self.dict_weights[i-1].L_structure/2)

            matrix_block.append(row_block)
            structure_block.append(struc_block)
            matrix_identity.append(row_identity_block)
            weight_adjustment_block.append(weights_adj_block)
            gradient_update_block.append(grad_update_block)

        self.deep_matrix_weights = csr_matrix(np.block(matrix_block))
        self.deep_matrix_structure = csr_matrix(np.block(structure_block))
        self.deep_matrix_identity = csr_matrix(np.block(matrix_identity))
        self.weights_adjustment_matrix = csr_matrix(np.block(weight_adjustment_block))
        self.weights_update_matrix = csr_matrix(np.block(gradient_update_block))
    
    def create_g_vec(self):
        vec_g = np.zeros((np.sum(self.dimensions),))
        for i in range(1, self.layers+1):
            end_range = np.sum(self.dimensions[:i+1])
            start_range = np.sum(self.dimensions[:i])
            vec_g[start_range:end_range] = self.tanh_factors[i-1]
        self.g_vec = vec_g[self.channels*self.image_dim**2:]

    def create_mult_vec(self):
        vec_mult = np.ones((np.sum(self.dimensions),))
        for i in range(1, self.layers+1):
            end_range = np.sum(self.dimensions[:i+1])
            start_range = np.sum(self.dimensions[:i])
            vec_mult[start_range:end_range] = self.mult_factors[i-1]
        self.mult_vec = vec_mult[self.channels*self.image_dim**2:]
    
    def activation_function(self, vec):
        return np.tanh(self.g_vec*vec)
    
    def neural_dynamics(self, img):
        conversion_ticker = 0
        x = img.flatten()
        u_vec = np.zeros(np.sum(self.dimensions))
        r_vec = np.zeros(np.sum(self.dimensions))
        r_vec[:self.channels*self.image_dim**2] = x
        delta = [np.inf]*self.layers
        W_tilda = self.deep_matrix_weights.multiply(self.deep_matrix_structure)+self.deep_matrix_identity
        updates = 0
        while updates < 3000:
            if all(ele < 1e-4 for ele in delta):
                conversion_ticker=1
                break
            lr = max((self.euler_step/(1+0.005*updates)), 0.05)
            delta_u = (-u_vec + W_tilda.dot(r_vec))[self.channels*self.image_dim**2:]
            u_vec[self.channels*self.image_dim**2:] += lr*delta_u
            r_vec[self.channels*self.image_dim**2:] = self.activation_function(u_vec[self.channels*self.image_dim**2:])
            updates += 1
            for layer in range(1, self.layers+1):
                start_token_large = np.sum(self.dimensions[:layer])
                end_token_large = np.sum(self.dimensions[:layer+1])
                start_token_small = int(np.sum(self.dimensions[1:][:layer-1]))
                end_token_small = np.sum(self.dimensions[1:][:layer])
                delta_layer = np.linalg.norm(delta_u[start_token_small:end_token_small])/np.linalg.norm(u_vec[start_token_large:end_token_large])
                delta[layer-1] = delta_layer  
        return r_vec, conversion_ticker
    
    def update_weights(self, r_vec):
        self.current_lr = max(self.lr/(1+self.decay*self.epoch), self.lr_floor)
        update_matrix = np.outer(r_vec, r_vec)
        grad_weights = self.weights_update_matrix.multiply(update_matrix - self.weights_adjustment_matrix.multiply(self.deep_matrix_weights))
        self.deep_matrix_weights += self.lr*grad_weights
                
    def training(self, epochs, images):
        self.n_images = images.shape[0]
        for epoch in range(epochs):
            img_array = shuffle(images, random_state = epoch)
            epoch_start = time.time()
            sum_ticker = 0
            for img in img_array:
                r, conversion_ticker = self.neural_dynamics(img)
                sum_ticker += conversion_ticker
                self.update_weights(r)
            self.epoch+=1
            epoch_end = time.time()
            epoch_time = epoch_end-epoch_start
            self.conversion_tickers.append(sum_ticker/self.n_images)
            print('Epoch: {0}\nTime_Taken: {1}\nConversion: {2}\nCurrent Learning Rate: {3}\n\n'.format(self.epoch, epoch_time, self.conversion_tickers[-1], self.current_lr))
            
class deep_network_GPU(object):
    def __init__(self, image_dim, channels, NpSs, strides, distances, 
                 layers, gamma, lr, lr_floor, decay, distances_lateral, tanh_factors, mult_factors, euler_step):
        self.image_dim = image_dim
        self.channels = channels
        self.NpSs = NpSs
        self.strides = strides
        self.distances = distances
        self.lateral_distances = distances_lateral
        self.layers = layers
        self.gamma = gamma
        self.lr = lr
        self.lr_floor = lr_floor
        self.current_lr = None
        self.decay = decay
        self.conversion_tickers = []
        self.costs = []
        self.epoch=0
        self.structure = None
        self.deep_matrix_weights = None
        self.deep_matrix_structure = None
        self.deep_matrix_identity = None
        self.weights_adjustment_matrix = None
        self.weights_update_matrix = None
        self.grad_matrix = None
        self.n_images = None
        self.dict_weights = {}
        self.dimensions = []
        self.g_vec = None
        self.mult_vec = None
        self.euler_step = euler_step
        self.tanh_factors = tanh_factors
        self.mult_factors = mult_factors
        self.W_gpu = None
        deep_network_GPU.create_deep_network(self)
    
    def create_deep_network(self):
        for i in range(self.layers+1):
            dim = int(np.prod(self.strides[:i]))
            self.dimensions.append(int((self.image_dim/dim)**2)*([self.channels]+self.NpSs)[i])
        
        for i in range(self.layers):
            layer_input_dim = int(self.image_dim/np.prod(self.strides[:i]))
            self.dict_weights[i]=network_weights(NpS=([self.channels]+self.NpSs)[i+1], distance_parameter=self.distances[i], 
                                                input_dim=layer_input_dim,
                                                stride = self.strides[i], previous_NpS = ([self.channels]+self.NpSs)[i], lateral_distance=self.lateral_distances[i])
            self.dict_weights[i].create_weights_matrix()
        
        matrix_block = []
        structure_block = []
        matrix_identity = []
        weight_adjustment_block = []
        gradient_update_block = []
        abs_structure_block = []

        for i, ele_row in enumerate(self.dimensions):
            row_block = []
            struc_block = []
            row_identity_block = []
            weights_adj_block = []
            grad_update_block = []
            abs_struc_block = []

            start_block = max(i-1, 0)
            end_block = max(len(self.dimensions)-start_block-3, 0)

            if i == 0:
                row_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                struc_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                row_identity_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                weights_adj_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                grad_update_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                abs_struc_block.append(np.zeros((ele_row, np.sum(self.dimensions))))

            elif i < len(self.dimensions)-1:
                if start_block > 0:
                    row_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    row_identity_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    weights_adj_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    grad_update_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    abs_struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))

                row_block.append(self.dict_weights[i-1].W)
                row_block.append(self.dict_weights[i-1].L)
                row_block.append(self.dict_weights[i].W.T)
                
                struc_block.append(self.dict_weights[i-1].W_structure/self.mult_factors[i-1])
                struc_block.append(-self.dict_weights[i-1].L_structure)
                struc_block.append(self.gamma*self.mult_factors[i]*self.dict_weights[i].W_structure.T)

                abs_struc_block.append(self.dict_weights[i-1].W_structure)
                abs_struc_block.append(self.dict_weights[i-1].L_structure)
                abs_struc_block.append(self.dict_weights[i].W_structure.T)
                
                row_identity_block.append(np.zeros((self.dict_weights[i-1].W_structure.shape)))
                row_identity_block.append(np.identity(self.dict_weights[i-1].L_structure.shape[0]))
                row_identity_block.append(np.zeros((self.dict_weights[i].W_structure.T.shape)))

                weights_adj_block.append(self.dict_weights[i-1].W_structure)
                weights_adj_block.append(self.dict_weights[i-1].L_structure/(1+self.gamma))
                weights_adj_block.append(self.dict_weights[i].W_structure.T)

                grad_update_block.append(self.dict_weights[i-1].W_structure)
                grad_update_block.append(self.dict_weights[i-1].L_structure/2)
                grad_update_block.append(self.dict_weights[i].W_structure.T)

                if end_block > 0:
                    row_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    row_identity_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    weights_adj_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    grad_update_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    abs_struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))

            elif i+1 == len(self.dimensions):
                if start_block > 0:
                    row_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    row_identity_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    weights_adj_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    grad_update_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    abs_struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))

                row_block.append(self.dict_weights[i-1].W)
                row_block.append(self.dict_weights[i-1].L)
                
                struc_block.append(self.dict_weights[i-1].W_structure/self.mult_factors[i-1])
                struc_block.append(-self.dict_weights[i-1].L_structure)

                abs_struc_block.append(self.dict_weights[i-1].W_structure)
                abs_struc_block.append(self.dict_weights[i-1].L_structure)
                
                row_identity_block.append(np.zeros((self.dict_weights[i-1].W_structure.shape)))
                row_identity_block.append(np.identity(self.dict_weights[i-1].L_structure.shape[0]))
                
                weights_adj_block.append(self.dict_weights[i-1].W_structure)
                weights_adj_block.append(self.dict_weights[i-1].L_structure)
                
                grad_update_block.append(self.dict_weights[i-1].W_structure)
                grad_update_block.append(self.dict_weights[i-1].L_structure/2)

            matrix_block.append(row_block)
            structure_block.append(struc_block)
            matrix_identity.append(row_identity_block)
            weight_adjustment_block.append(weights_adj_block)
            gradient_update_block.append(grad_update_block)
            abs_structure_block.append(abs_struc_block)

        self.deep_matrix_weights = cp.asarray(np.block(matrix_block))
        self.deep_matrix_structure = cp.asarray(np.block(structure_block))
        self.deep_matrix_identity = cp.asarray(np.block(matrix_identity))
        self.weights_adjustment_matrix = cp.asarray(np.block(weight_adjustment_block))
        self.weights_update_matrix = cp.asarray(np.block(gradient_update_block))
        self.structure = cp.asarray(np.block(abs_structure_block))
    
    def activation_function(self, vec):
        return cp.tanh(vec)
    
    def neural_dynamics(self, img):
        conversion_ticker = 0
        x = img.flatten()
        u_vec = cp.asarray(np.zeros(np.sum(self.dimensions)))
        r_vec = np.zeros(np.sum(self.dimensions))
        r_vec[:self.channels*self.image_dim**2] = x
        r_vec = cp.asarray(r_vec)
        delta = [cp.inf]*self.layers
        self.W_gpu = csr_gpu(self.deep_matrix_weights*self.deep_matrix_structure + self.deep_matrix_identity)
        updates = 0
        while updates < 3000:
            if all(ele < 1e-4 for ele in delta):
                conversion_ticker=1
                break
            lr = max((self.euler_step/(1+0.005*updates)), 0.05)
            delta_u = -u_vec + self.W_gpu.dot(r_vec)
            u_vec[self.channels*self.image_dim**2:] += lr*delta_u[self.channels*self.image_dim**2:]
            r_vec[self.channels*self.image_dim**2:] = self.activation_function(u_vec[self.channels*self.image_dim**2:])
            updates += 1
            if (updates+1)%100 == 0:
                for layer in range(1, self.layers+1):
                    start_token_large = np.sum(self.dimensions[:layer])
                    end_token_large = np.sum(self.dimensions[:layer+1])
                    start_token_small = int(np.sum(self.dimensions[1:][:layer-1]))
                    end_token_small = np.sum(self.dimensions[1:][:layer])
                    delta_layer = cp.linalg.norm(delta_u[start_token_small:end_token_small])/cp.linalg.norm(u_vec[start_token_large:end_token_large])
                    delta[layer-1] = delta_layer  
        return r_vec, conversion_ticker
    
    def update_weights(self, r_vec):
        self.current_lr = max(self.lr/(1+self.decay*self.epoch), self.lr_floor)
        #r_vec = cp.asnumpy(r_vec)
        update_matrix = cp.outer(r_vec, r_vec)
        grad_weights = self.weights_update_matrix*(update_matrix - self.weights_adjustment_matrix*self.deep_matrix_weights)
        self.deep_matrix_weights += self.current_lr*grad_weights
                
    def training(self, epochs, images):
        self.n_images = images.shape[0]
        for epoch in range(epochs):
            img_array = shuffle(images, random_state = epoch)
            epoch_start = time.time()
            sum_ticker = 0
            for img in img_array:
                r, conversion_ticker = self.neural_dynamics(img)
                sum_ticker += conversion_ticker
                self.update_weights(r)
            self.epoch+=1
            epoch_end = time.time()
            epoch_time = epoch_end-epoch_start
            self.conversion_tickers.append(sum_ticker/self.n_images)
            print('Epoch: {0}\nTime_Taken: {1}\nConversion: {2}\nCurrent Learning Rate: {3}\n\n'.format(self.epoch, epoch_time, self.conversion_tickers[-1], self.current_lr))
