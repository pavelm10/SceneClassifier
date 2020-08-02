from tensorflow import keras
import keras.backend as K
import tensorflow as tf


class AccumAdadelta(keras.optimizers.Optimizer):
    def __init__(self, lr=1.0, rho=0.95, epsilon=None, decay=0., accum_iters=1,
                 **kwargs):
        if accum_iters < 1:
            raise ValueError('accum_iters must be >= 1')
        super(AccumAdadelta, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        if epsilon is None:
            epsilon = K.epsilon()
        self.rho = rho
        self.epsilon = epsilon
        self.initial_decay = decay
        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))
        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        delta_accumulators = [K.zeros(shape) for shape in shapes]
        gs = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators + delta_accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        completed_updates = K.cast(tf.floordiv(self.iterations, self.accum_iters), K.floatx())
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * completed_updates))

        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
        update_switch = K.cast(update_switch, K.floatx())

        for p, g, a, d_a, tg in zip(params, grads, accumulators, delta_accumulators, gs):
            # update accumulator
            sum_grad = tg + g
            avg_grad = sum_grad / self.accum_iters_float
            new_a = self.rho * a + (1. - self.rho) * K.square(avg_grad)
            self.updates.append(K.update(a, (1 - update_switch) * a + update_switch * new_a))

            # use the new accumulator and the *old* delta_accumulator
            update = avg_grad * K.sqrt(d_a + self.epsilon) / K.sqrt(new_a + self.epsilon)
            new_p = p - lr * update

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * K.square(update)
            self.updates.append(K.update(d_a, (1 - update_switch) * d_a + update_switch * new_d_a))
            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))

        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': self.rho,
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(AccumAdadelta, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
